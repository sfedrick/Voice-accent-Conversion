# FACsimile

Speech-to-Speech Voice Conversion for Accent Localization

## Members
Tyler Durkin, Shaun Fedrick, and Anthony Santana

## Background
We found that many speech centric models focus on either text-to-speech (TTS) or speech-to-text (STT) conversion. When it comes to generative speech, the TTS approach is the most common with research focusing on making the generated speech as realistic as possible. In contrast, we have elected to focus on speech-to-speech (S2S) voice conversion (VC), a topic that has historically had less research but seen a flurry of activity in recent years. We are interested in using deep learning models to convert spoken audio from one person’s voice to another person’s voice directly, without any intermediate STT-TTS conversion. To a large degree this can be seen as an application of style transfer to the audio domain.

The use case we intend to focus on is voice conversion for accent localization. This task has a multitude of applications including addressing communication challenges, supporting language learners and improving speech-based interfaces. For context, studies show that speakers conversing with speakers with foreign accents have more difficulty understanding and reduced comprehension even in ideal listening scenarios. Other studies have found reduced performance in automatic speech recognition systems when used by non-native English speakers. Lastly, this type of model could be used to retain the source speaker’s characteristics or apply native accents during S2S language translation.

This task involves considerable architectural deliberation, such as determining how to transform vocal features of an audio sample such as timbre, intonation, and prosody to produce a piece of audio with the desired effect, deciding whether to use traditional voice conversion with parallel datasets or integrate more novel methods with non-parallel data usage, etc. resulting in a project that is both challenging and rewarding.

We acknowledge the ethical concerns surrounding voice cloning and adjacent technologies including the need for consent, their impact on privacy, and potential for misuse. Our goal for this project, however, is not to outright clone a voice but rather to copy the accent characteristics of a reference speaker onto a source speaker. This presents its own ethical concerns, but also has many beneficial applications and ultimately the onus for responsible use is placed on the user.

## Approach
We intend to implement an encoder-decoder architecture inspired by AdaptVC that takes in an audio file and outputs another audio file with the same content, but with variations to its voice that are recognized as some target accent. Currently, we are considering using a model like HuBERT, a self-supervised learning (SSL) model for speech processing with the capacity to extract linguistic features and speaker attributes, as reference for our encoder. Our decoder under consideration is a transformer-based U-Net architecture with cross-attention layers to align the linguistic content with the target speaker embeddings. Other models we are interested in experimenting with include WavLM, HiFi-GAN, and SoftStream.

It should be noted that our plan is not to design and train a brand new model from scratch, but rather to leverage existing models, assemble them in a novel way to meet our project goals, and fine tune them on additional data for foreign accent conversion. Experimentation will be focused on incremental architectural improvements and/or hyperparameter tuning to improve results and our report will include documenting different architectures and relative perceived success.

Common evaluation metrics for generative speech models are subjective scores based on the Mean Opinion Score (MOS) such as Naturalness MOS (MOS-N) and Similarity MOS (MOS-S). Obtaining such scores would require a large survey with multiple outputs from different models and is  outside the scope of this project.  However, we are still investigating methods for objectively evaluating output such as ASV scores for separate voice characteristics, Mel Cepstral Distortion (MCD), or Short-Time Objective Intelligibility (STOI).


## Related Work
State of the art solutions for general voice conversion include AdaptVC, FreeVC, HiFi-VC and WavThruVec. Each has its own strengths and weaknesses, such AdaptVC's superiority in zero-shot voice conversion and FreeVC's robustness against low-quality source speech. State of the art solutions for foreign accent conversion include Accentron and Nechaev and Kosyakov's model and will serve as a baseline for our own model’s performance.

Additional research we have found that may help address our work on foreign accent conversion include PolyVoice, and S2S translation system that aims to “preserve the voice characteristics and speaking style of the original speech,” and WavLM-pro, a fine-tuned version of WavLM created by Orange Telecom that “produces embeddings that globally represent the non-timbral trains (prosody, accent, [etc.]) of a speaker’s voice” as part of their research into voice conversion and anonymization.


## Datasets
- GLOBE: a curated and enhanced subset of the Common Voice English corpus with worldwide accents designed for zero-shot adaptive TTS systems. GLOBE is derived from Common Voice, a crowdsourced open speech dataset by Mozilla, but focuses on improved speech quality through rigorous filtering and enhancement.
- Speech Accent Archive: a dataset of 2,140 speech samples with speakers from 177 countries who have 214 different native languages. It was used to train Accentron.
- Voxpopuli: a multilingual speech corpus collected from European Parliament event recordings that contains 29 hours of transcribed speech data of non-native English intended for research in ASR for accented speech.
- CMU ARCTIC and L2-ARCTIC: A speech database established for research in speech synthesis. It contains phonetically balanced sentences recorded by single speakers, with some accent diversity in US, Canadian, Scottish and Indian English accents. L2-ARCTIC expands on CMU ARCTIC with a speech corpus of 24 non-native speakers whose first languages include Hindi, Korean, Arabic, and others, each recording speech from CMU's prompts. These models were used to train Nechaev and Kosyakov's model and were used to evaluate Accentron.
- VCTK - A corpus of speech data developed by the University of Edinburgh's CSTR with samples from 110 English speakers with various accents, covering about 400 sentences per speaker. Referenced in Google DeepMind's WaveNet and was used to train Nechaev and Kosyakov's model.
