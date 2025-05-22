import asyncio
import base64
import time
from functools import partial
from typing import Any, Awaitable, Callable, Mapping, Optional

import aiohttp
from daily.daily import AudioData
from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import create_default_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    OutputImageRawFrame,
    StartFrame,
    StartInterruptionFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.services.daily import (
    DailyCallbacks,
    DailyParams,
    DailyTransportClient,
)


class TavusApi:
    BASE_URL = "https://tavusapi.com/v2"

    def __init__(self, api_key: str, session: aiohttp.ClientSession):
        self._api_key = api_key
        self._session = session
        self._headers = {"Content-Type": "application/json", "x-api-key": self._api_key}

    async def create_conversation(self, replica_id: str, persona_id: str) -> dict:
        url = f"{self.BASE_URL}/conversations"
        payload = {
            "replica_id": replica_id,
            "persona_id": persona_id,
        }
        async with self._session.post(url, headers=self._headers, json=payload) as r:
            r.raise_for_status()
            response = await r.json()
            logger.debug(f"Created Tavus conversation: {response}")
            return response

    async def end_conversation(self, conversation_id: str):
        if conversation_id is None:
            return

        url = f"{self.BASE_URL}/conversations/{conversation_id}/end"
        async with self._session.post(url, headers=self._headers) as r:
            r.raise_for_status()
            logger.debug(f"Ended Tavus conversation {conversation_id}")

    async def get_persona_name(self, persona_id: str) -> str:
        url = f"{self.BASE_URL}/personas/{persona_id}"
        async with self._session.get(url, headers=self._headers) as r:
            r.raise_for_status()
            response = await r.json()
            logger.debug(f"Fetched Tavus persona: {response}")
            return response["persona_name"]


class TavusCallbacks(BaseModel):
    """Callback handlers for Daily events.

    Attributes:
        on_participant_joined: Called when a participant joins.
        on_participant_left: Called when a participant leaves.
    """

    on_participant_joined: Callable[[Mapping[str, Any]], Awaitable[None]]
    on_participant_left: Callable[[Mapping[str, Any], str], Awaitable[None]]


class TavusTransportClient:
    def __init__(
        self,
        *,
        bot_name: str,
        params: DailyParams,
        callbacks: TavusCallbacks,
        api_key: str,
        replica_id: str,
        persona_id: str = "pipecat0",  # Use `pipecat0` so that your TTS voice is used in place of the Tavus persona
        session: aiohttp.ClientSession,
        sample_rate,
    ) -> None:
        self._bot_name = bot_name
        self._api = TavusApi(api_key, session)
        self._replica_id = replica_id
        self._persona_id = persona_id
        self._conversation_id: Optional[str] = None
        self._other_participant_has_joined = False
        self._client: Optional[DailyTransportClient] = None
        self._callbacks = callbacks
        self._sample_rate = sample_rate if sample_rate is not None else 16000
        self._setup_in_progress = False
        self._params = params

    async def initialize(self) -> str:
        response = await self._api.create_conversation(self._replica_id, self._persona_id)
        self._conversation_id = response["conversation_id"]
        return response["conversation_url"]

    async def setup(self, setup: FrameProcessorSetup):
        if self._setup_in_progress or self._conversation_id is not None:
            return
        self._setup_in_progress = True
        try:
            room_url = await self.initialize()
            daily_callbacks = DailyCallbacks(
                on_active_speaker_changed=partial(
                    self._on_handle_callback, "on_active_speaker_changed"
                ),
                on_joined=self._on_joined,
                on_left=self._on_left,
                on_error=partial(self._on_handle_callback, "on_error"),
                on_app_message=partial(self._on_handle_callback, "on_app_message"),
                on_call_state_updated=partial(self._on_handle_callback, "on_call_state_updated"),
                on_client_connected=partial(self._on_handle_callback, "on_client_connected"),
                on_client_disconnected=partial(self._on_handle_callback, "on_client_disconnected"),
                on_dialin_connected=partial(self._on_handle_callback, "on_dialin_connected"),
                on_dialin_ready=partial(self._on_handle_callback, "on_dialin_ready"),
                on_dialin_stopped=partial(self._on_handle_callback, "on_dialin_stopped"),
                on_dialin_error=partial(self._on_handle_callback, "on_dialin_error"),
                on_dialin_warning=partial(self._on_handle_callback, "on_dialin_warning"),
                on_dialout_answered=partial(self._on_handle_callback, "on_dialout_answered"),
                on_dialout_connected=partial(self._on_handle_callback, "on_dialout_connected"),
                on_dialout_stopped=partial(self._on_handle_callback, "on_dialout_stopped"),
                on_dialout_error=partial(self._on_handle_callback, "on_dialout_error"),
                on_dialout_warning=partial(self._on_handle_callback, "on_dialout_warning"),
                on_participant_joined=self._callbacks.on_participant_joined,
                on_participant_left=self._callbacks.on_participant_left,
                on_participant_updated=partial(self._on_handle_callback, "on_participant_updated"),
                on_transcription_message=partial(
                    self._on_handle_callback, "on_transcription_message"
                ),
                on_recording_started=partial(self._on_handle_callback, "on_recording_started"),
                on_recording_stopped=partial(self._on_handle_callback, "on_recording_stopped"),
                on_recording_error=partial(self._on_handle_callback, "on_recording_error"),
            )
            self._client = DailyTransportClient(
                room_url, None, "Pipecat", self._params, daily_callbacks, self._bot_name
            )
            await self._client.setup(setup)
        except Exception as e:
            logger.error(f"Failed to setup TavusTransportClient: {e}")
            await self._api.end_conversation(self._conversation_id)
        finally:
            self._setup_in_progress = False

    async def cleanup(self):
        if self._client is None:
            return
        await self._client.cleanup()
        self._client = None

    async def _on_joined(self, data):
        logger.debug("TavusTransportClient joined!")

    async def _on_left(self):
        logger.debug("TavusTransportClient left!")

    async def _on_handle_callback(self, event_name, *args, **kwargs):
        logger.trace(f"[Callback] {event_name} called with args={args}, kwargs={kwargs}")

    async def get_persona_name(self) -> str:
        return await self._api.get_persona_name(self._persona_id)

    async def start(self, frame: StartFrame):
        logger.debug("TavusTransportClient start invoked!")
        await self._client.start(frame)
        await self._client.join()

    async def stop(self):
        await self._client.leave()
        await self._api.end_conversation(self._conversation_id)

    async def capture_participant_video(
        self,
        participant_id: str,
        callback: Callable,
        framerate: int = 30,
        video_source: str = "camera",
        color_format: str = "RGB",
    ):
        await self._client.capture_participant_video(
            participant_id, callback, framerate, video_source, color_format
        )

    async def capture_participant_audio(
        self,
        participant_id: str,
        callback: Callable,
        audio_source: str = "microphone",
    ):
        await self._client.capture_participant_audio(participant_id, callback, audio_source)

    async def send_message(self, frame: TransportMessageFrame | TransportMessageUrgentFrame):
        await self._client.send_message(frame)

    @property
    def out_sample_rate(self) -> int:
        return self._client.out_sample_rate

    async def encode_audio_and_send(self, audio: bytes, done: bool, inference_id: str):
        """Encodes audio to base64 and sends it to Tavus"""
        audio_base64 = base64.b64encode(audio).decode("utf-8")
        await self._send_audio_message(audio_base64, done=done, inference_id=inference_id)

    async def send_interrupt_message(self) -> None:
        transport_frame = TransportMessageUrgentFrame(
            message={
                "message_type": "conversation",
                "event_type": "conversation.interrupt",
                "conversation_id": self._conversation_id,
            }
        )
        await self.send_message(transport_frame)

    async def _send_audio_message(self, audio_base64: str, done: bool, inference_id: str):
        transport_frame = TransportMessageUrgentFrame(
            message={
                "message_type": "conversation",
                "event_type": "conversation.echo",
                "conversation_id": self._conversation_id,
                "properties": {
                    "modality": "audio",
                    "inference_id": inference_id,
                    "audio": audio_base64,
                    "done": done,
                    "sample_rate": self._sample_rate,
                },
            }
        )
        await self.send_message(transport_frame)

    async def update_subscriptions(self, participant_settings=None, profile_settings=None):
        await self._client.update_subscriptions(
            participant_settings=participant_settings, profile_settings=profile_settings
        )

    async def write_raw_audio_frames(self, frames: bytes, destination: Optional[str] = None):
        await self._client.write_raw_audio_frames(frames, destination)


class TavusInputTransport(BaseInputTransport):
    def __init__(
        self,
        client: TavusTransportClient,
        params: TransportParams,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._client = client
        self._params = params
        self._resampler = create_default_resampler()

    async def setup(self, setup: FrameProcessorSetup):
        await super().setup(setup)
        await self._client.setup(setup)

    async def cleanup(self):
        await super().cleanup()
        await self._client.cleanup()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._client.start(frame)
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._client.stop()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._client.stop()

    async def start_capturing_audio(self, participant):
        if self._params.audio_in_enabled:
            logger.info(
                f"TavusTransportClient start capturing audio for participant {participant['id']}"
            )
            await self._client.capture_participant_audio(
                participant["id"], self._on_participant_audio_data
            )

    async def _on_participant_audio_data(
        self, participant_id: str, audio: AudioData, audio_source: str
    ):
        resampled = await self._resampler.resample(
            audio.audio_frames, audio.sample_rate, self._client.out_sample_rate
        )
        frame = InputAudioRawFrame(
            audio=resampled,
            sample_rate=self._client.out_sample_rate,
            num_channels=audio.num_channels,
        )
        frame.transport_source = audio_source
        await self.push_audio_frame(frame)


class TavusOutputTransport(BaseOutputTransport):
    def __init__(
        self,
        client: TavusTransportClient,
        params: TransportParams,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._client = client
        self._params = params
        self._samples_sent = 0
        self._start_time = time.time()

    async def setup(self, setup: FrameProcessorSetup):
        await super().setup(setup)
        await self._client.setup(setup)

    async def cleanup(self):
        await super().cleanup()
        await self._client.cleanup()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._samples_sent = 0
        self._start_time = time.time()
        await self._client.start(frame)
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._client.stop()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._client.stop()

    async def send_message(self, frame: TransportMessageFrame | TransportMessageUrgentFrame):
        logger.info(f"TavusOutputTransport sending message {frame}")
        await self._client.send_message(frame)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartInterruptionFrame):
            await self._handle_interruptions()
        elif isinstance(frame, TTSStartedFrame):
            self._current_idx_str = str(frame.id)
        elif isinstance(frame, TTSStoppedFrame):
            logger.debug(f"TAVUS: {self}: stopped speaking")
            await self._client.encode_audio_and_send(b"\x00\x00", True, self._current_idx_str)

    async def _handle_interruptions(self):
        await self._client.send_interrupt_message()

    async def write_raw_audio_frames(self, frames: bytes, destination: Optional[str] = None):
        # Compute wait time for synchronization
        wait = self._start_time + (self._samples_sent / self._sample_rate) - time.time()
        if wait > 0:
            await asyncio.sleep(wait)

        await self._client.encode_audio_and_send(frames, False, self._current_idx_str)

        # Update timestamp based on number of samples sent
        self._samples_sent += len(frames) // 2  # 2 bytes per sample (16-bit)

    async def write_raw_video_frame(
        self, frame: OutputImageRawFrame, destination: Optional[str] = None
    ):
        pass


class TavusTransport(BaseTransport):
    """Transport implementation for Tavus video calls."""

    def __init__(
        self,
        bot_name: str,
        session: aiohttp.ClientSession,
        api_key: str,
        replica_id: str,
        persona_id: str = "pipecat0",  # Use `pipecat0` so that your TTS voice is used in place of the Tavus persona
        params: DailyParams = DailyParams(),
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        super().__init__(input_name=input_name, output_name=output_name)
        self._params = params

        # TODO: Filipi - We can remove this if we stop sending the audio through app messages
        # Limiting this so we don't go over 20 messages per second
        # each message is going to have 50ms of audio
        self._params.audio_out_10ms_chunks = 5

        callbacks = TavusCallbacks(
            on_participant_joined=self._on_participant_joined,
            on_participant_left=self._on_participant_left,
        )
        self._client = TavusTransportClient(
            bot_name="Pipecat",
            callbacks=callbacks,
            api_key=api_key,
            replica_id=replica_id,
            persona_id=persona_id,
            session=session,
            sample_rate=params.audio_out_sample_rate,
            params=params,
        )
        self._input: Optional[TavusInputTransport] = None
        self._output: Optional[TavusOutputTransport] = None
        self._tavus_participant_id = None

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")

    async def _on_participant_left(self, participant, reason):
        persona_name = await self._client.get_persona_name()
        if participant.get("info", {}).get("userName", "") != persona_name:
            await self._on_client_disconnected(participant)

    async def _on_participant_joined(self, participant):
        # get persona, look up persona_name, set this as the bot name to ignore
        persona_name = await self._client.get_persona_name()
        # Ignore the Tavus replica's microphone
        if participant.get("info", {}).get("userName", "") == persona_name:
            self._tavus_participant_id = participant["id"]
        else:
            await self._on_client_connected(participant)
            if self._tavus_participant_id:
                logger.debug(f"Ignoring {self._tavus_participant_id}'s microphone")
                await self.update_subscriptions(
                    participant_settings={
                        self._tavus_participant_id: {
                            "media": {"microphone": "unsubscribed"},
                        }
                    }
                )
            if self._input:
                await self._input.start_capturing_audio(participant)

    async def update_subscriptions(self, participant_settings=None, profile_settings=None):
        await self._client.update_subscriptions(
            participant_settings=participant_settings,
            profile_settings=profile_settings,
        )

    def input(self) -> FrameProcessor:
        if not self._input:
            self._input = TavusInputTransport(client=self._client, params=self._params)
        return self._input

    def output(self) -> FrameProcessor:
        if not self._output:
            self._output = TavusOutputTransport(client=self._client, params=self._params)
        return self._output

    async def _on_client_connected(self, participant: Any):
        await self._call_event_handler("on_client_connected", participant)

    async def _on_client_disconnected(self, participant: Any):
        await self._call_event_handler("on_client_disconnected", participant)
