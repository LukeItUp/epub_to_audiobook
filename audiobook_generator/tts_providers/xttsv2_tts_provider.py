import logging
import os
import math
import io

import numpy as np
import soundfile
from pydub import AudioSegment
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core.utils import split_text, set_audio_tags
from audiobook_generator.tts_providers.base_tts_provider import BaseTTSProvider


logger = logging.getLogger(__name__)
XTTS_MODEL_PATH = "/home/luke/Projects/XTTS-v2"
SPEAKER_PATH = os.path.join(XTTS_MODEL_PATH, "samples/en_sample.wav")


def get_supported_formats():
    return ["mp3"]  # TODO: figure out more supported formats


class XTTSV2TTSProvider(BaseTTSProvider):
    def __init__(self, config: GeneralConfig):
        logger.setLevel(config.log)
        # TTS provider specific config
        self.xtts_config = XttsConfig()
        xtts_config_path = os.path.join(XTTS_MODEL_PATH, "config.json")
        self.xtts_config.load_json(xtts_config_path)
        self.model = Xtts.init_from_config(self.xtts_config)
        self.model.load_checkpoint(
            self.xtts_config,
            checkpoint_dir=XTTS_MODEL_PATH,
            eval=True
        )
        if config.gpu:
            self.model.cuda()

        # 0.000$ per 1 million characters
        # or 0.000$ per 1000 characters
        self.price = 0.000        
        super().__init__(config)

    def __str__(self) -> str:
        return f"{self.config}"

    def validate_config(self):
        # TODO
        pass
    
    def text_to_speech(
        self,
        text: str,
        output_file: str,
        audio_tags: AudioTags,
    ):
        self.file_data = np.array([], dtype=np.float32)
        self._chunkify(text)

        logger.debug(f"Exporting the audio")
        tmp_file = output_file.replace(".mp3","_tmp.wav")
        soundfile.write(tmp_file, self.file_data, 24000)

        logger.debug(f"Converting to mp3")
        AudioSegment.from_wav(tmp_file).export(output_file)
        os.remove(tmp_file)

        set_audio_tags(output_file, audio_tags)
        logger.info(f"Saved the audio to: {output_file}")

    def estimate_cost(self, total_chars):
        return math.ceil(total_chars / 1000) * self.price

    def get_break_string(self):
        return "@BRK#"
    
    def get_output_file_extension(self):
        return "mp3"
        # if self.config.output_format.endswith("wav"):
        #     return "wav"
        # else:
        #     # Only wav supported for now (?)
        #     raise NotImplementedError(
        #         f"Unknown file extension for output format: {self.config.output_format}. Only wav supported in xttsv2-tts."
        #     )

    def _parse_text(self, text):
        logger.debug(f"Parsing the text, looking for break/pauses in text: <{text}>")
        if self.get_break_string() not in text:
            logger.debug(f"No break/pauses found in the text")
            return [text]

        parts = text.split(self.get_break_string())
        logger.debug(f"split into <{len(parts)}> parts: {parts}")
        return parts

    def _generate_pause(self, time: int):
        logger.debug(f"Generating pause")
        # pause time should be provided in ms
        n = 24 * time
        silence = [0.0 for _ in range(n)]
        return silence

    def _generate_audio(self, text: str):
        logger.debug(f"Generation audio for: <{text}>")
        output = self.model.synthesize(
            text,
            self.xtts_config,
            speaker_wav=SPEAKER_PATH,
            gpt_cond_len=3,
            language="en",
        ).get("wav")
        return output

    def _chunkify(self, full_text):
        logger.debug("Chunkifying the text")
        parsed_text = self._parse_text(full_text)
        for content in parsed_text:
            logger.debug(f"Content from parsed: <{content}>")
            remainder = content
            while remainder:
                digest_chunk, remainder = self._xtts_digest_chunk(remainder)
                audio_data = self._generate_audio(digest_chunk)
                self.file_data = np.append(self.file_data, audio_data)
                
            if content != parsed_text[-1]:
                pause_data = self._generate_pause(1250)
                self.file_data = np.append(self.file_data, pause_data)
        logger.debug("Chunkifying done")
        
    def _xtts_digest_chunk(self, text, max_len=250):
        if len(text) <= max_len:
            return text, ""

        first_part = text[:max_len]
        end_position = max(
            first_part.rfind("."),
            first_part.rfind("?"),
            first_part.rfind("!"),
        )

        if end_position == -1:
            # A very long sentence? -> spliting it into two parts
            end_position = first_part.rfind(" ")
        if end_position == -1:
            # A very long word? -> spliting it into two parts
            end_position = max_len
        else:
            end_position += 1

        first_part = text[:end_position]
        remainder = text[end_position:]
        return first_part, remainder