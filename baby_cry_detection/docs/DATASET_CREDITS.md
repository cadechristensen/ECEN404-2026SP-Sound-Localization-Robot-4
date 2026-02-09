# Dataset Credits and Attribution

This document provides proper attribution for all datasets and audio sources used in this baby cry detection project.

## Primary Datasets

### 1. Donate-a-Cry Corpus
- **Source**: https://github.com/gveres/donateacry-corpus
- **Files Used**: 696 baby cry samples in `data/cry/`
- **License**: Check repository for specific license terms
- **Description**: Collection of baby cry recordings from the Donate-a-Cry project
- **Citation**:
  ```
  Donate-a-Cry Corpus
  Available at: https://github.com/gveres/donateacry-corpus
  ```

### 2. mahmudulhasan01/baby_crying_sound (Hugging Face)
- **Source**: https://huggingface.co/datasets/mahmudulhasan01/baby_crying_sound
- **Files Used**: 69 unique baby cry samples in `data/cry/`
- **Original Total**: 1,197 files downloaded (1,128 removed as duplicates/mislabeled)
- **License**: Unknown (check Hugging Face dataset page)
- **Description**: Baby crying sound dataset with multiple cry categories
- **Quality Notes**:
  - Removed 696 duplicates matching existing Donate-a-Cry samples
  - Removed 216 mislabeled files (baby laughs and silence incorrectly labeled as cries)
  - Filtered 216 files matching non-cry categories
  - Final: 69 unique, verified cry samples retained

### 3. Baby Cry Sense Dataset (Kaggle)
- **Source**: https://www.kaggle.com/datasets/mennaahmed23/baby-cry-sense-dataset
- **Files Used**: 57 unique baby cry samples in `data/cry/`
- **Original Total**: 1,054 files downloaded (997 removed as duplicates)
- **License**: Check Kaggle dataset page
- **Formats**: .mp3 and .3gp files
- **Description**: Categorized baby cry dataset with types (hungry, belly pain, burping, discomfort, scared, tired, lonely, cold/hot)
- **Quality Notes**:
  - Removed 1,047 duplicates (977 from data/cry, 70 from previous additions)
  - Removed 22 files with "Copy" in filename (duplicates)
  - Final: 57 unique cry samples with diverse cry types
  - Prompted addition of .3gp format support to the model

### 4. ICSD (Infant Cry Sound Dataset)
- **Source**: Contact dataset authors for access (research dataset)
- **Files Used**: 2,312 real baby cry samples in `data/cry_ICSD/`
- **Composition**:
  - 424 strong label files (precise onset/offset timestamps)
  - 1,888 weak label files (clip-level annotations)
- **Files Excluded**:
  - 883 snoring files (not baby cries)
  - 500 synthetic audio files (synth_*: artificially mixed, not suitable for sound localization)
- **License**: Check with dataset maintainers
- **Format**: .wav files
- **Label Types**:
  - Strong labels: Precise temporal annotations (onset/offset times)
  - Weak labels: Clip-level annotations (cry present, no timing)
- **Description**: Professional dataset with real infant cry recordings. Both strong and weak labels are real recordings - the difference is annotation detail, not audio quality.
- **Purpose**: Binary classification and sound localization in household environments
- **Original Location**: `data/ICSD_cry/` (archived)

### 5. LibriSpeech ASR Corpus (train-clean-100)
- **Source**: https://www.openslr.org/12
- **Files Used**: 2,241 adult speech samples in `data/adult_speech/`
- **Original Total**: 4,481 files (reduced to 2,241 for better class balance)
- **License**: CC BY 4.0
- **Description**: Large-scale corpus of read English speech
- **Citation**:
  ```
  @inproceedings{panayotov2015librispeech,
    title={Librispeech: an ASR corpus based on public domain audio books},
    author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
    booktitle={2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    pages={5206--5210},
    year={2015},
    organization={IEEE}
  }
  ```

### 6. ESC-50: Dataset for Environmental Sound Classification
- **Source**: https://github.com/karolpiczak/ESC-50
- **Files Used**: 1,960 household/environmental sounds in `data/household/` and `data/noise/`
- **License**: CC BY-NC 3.0
- **Description**: Collection of 2000 environmental audio recordings (50 classes)
- **Citation**:
  ```
  @inproceedings{piczak2015esc,
    title={ESC: Dataset for environmental sound classification},
    author={Piczak, Karol J},
    booktitle={Proceedings of the 23rd ACM international conference on Multimedia},
    pages={1015--1018},
    year={2015}
  }
  ```

### 7. Freesound.org Community Sounds
- **Source**: https://freesound.org/
- **Files Used**: ~140 baby non-cry sounds in `data/baby_noncry/`
- **License**: Various Creative Commons licenses (individual files retain original licenses)
- **Description**: Community-contributed baby babbling, cooing, laughing, and giggling sounds
- **Attribution**: Individual file attributions below
- **API Documentation**: https://freesound.org/docs/api/
- **Search Terms Used**: baby babbling, baby cooing, baby giggle, baby laugh happy, infant babble, baby gurgle, baby happy sounds, baby play sounds

#### Freesound Individual File Credits
All files downloaded from Freesound.org are prefixed with `freesound_[ID]_` where ID is the Freesound sound ID.
To view individual file credits and licenses, visit: `https://freesound.org/people/[username]/sounds/[ID]/`

**Example files and their creators:**
- freesound_139046_Babbling - Happy 2.ogg - User: [View on Freesound](https://freesound.org/people/139046/)
- freesound_416639_Baby Gibberish 4wav.ogg - User: [View on Freesound](https://freesound.org/people/416639/)
- freesound_458645_babygigglemp3.ogg - User: [View on Freesound](https://freesound.org/people/458645/)

**Note**: Each Freesound file retains its original Creative Commons license. Please check individual file pages for specific license terms.

### 8. giulbia/baby_cry_detection Repository
- **Source**: https://github.com/giulbia/baby_cry_detection
- **Files Used**: 216 processed samples in `data/baby_noncry/`
  - 108 baby laugh samples (from '903 - Baby laugh')
  - 108 silence samples (from '901 - Silence')
- **License**: Check repository for specific license terms
- **Description**: Baby cry detection project with pre-processed audio samples
- **Note**: These files appear to originate from ESC-50 dataset
- **Consolidated Location**: Merged into `data/baby_noncry/` directory

### 9. CryCeleb2023 Dataset
- **Source**: CryCeleb2023 baby cry dataset
- **Files Used**: 1,423 baby cry samples in `data/cry_crycaleb/`
- **License**: Check dataset source for specific license terms
- **Description**: Baby cry recordings from the CryCeleb dataset
- **Format**: .wav files
- **Purpose**: Binary classification and sound localization
- **Citation**:
  ```
  CryCeleb2023 Dataset
  Used for baby cry detection and classification
  ```

## Copyright Notice

This project does not own the copyright to any of the audio files used for training.
Copyright remains with the original owners and creators of the audio content.

All datasets are used for research and educational purposes under their respective licenses.

## License Compliance

Please ensure you comply with the following license requirements when using this project:

1. **LibriSpeech (CC BY 4.0)**:
   - Attribution required
   - Free for commercial and non-commercial use
   - https://creativecommons.org/licenses/by/4.0/

2. **ESC-50 (CC BY-NC 3.0)**:
   - Attribution required
   - Non-commercial use only
   - https://creativecommons.org/licenses/by-nc/3.0/

3. **Freesound (Various CC licenses)**:
   - Check individual file licenses
   - Most are CC BY, CC BY-NC, or CC0
   - Attribution typically required

4. **Donate-a-Cry Corpus**:
   - Check repository for specific terms

## How to Cite This Project's Dataset

If you use this combined dataset, please cite all original sources listed above.

## Acknowledgments

We gratefully acknowledge:
- The Donate-a-Cry project for baby cry recordings
- mahmudulhasan01 for the Hugging Face baby crying sound dataset
- mennaahmed23 for the Kaggle Baby Cry Sense Dataset
- ICSD dataset creators and maintainers for the Infant Cry Sound Dataset
- CryCeleb2023 dataset contributors for baby cry recordings
- The LibriSpeech project and contributors for speech data
- Karol J. Piczak for the ESC-50 environmental sound dataset
- The Freesound.org community for user-contributed sounds
- giulbia for the baby cry detection repository and processed samples
- All individual Freesound contributors whose sounds we used

## Data Usage Statement

All audio files are used strictly for:
- Research purposes
- Educational purposes
- Development of baby cry detection algorithms
- Non-commercial applications

If you intend to use this project or its trained models commercially, please verify compliance with all dataset licenses, particularly ESC-50's non-commercial restriction.

## Questions or Concerns

If you are a copyright holder and have concerns about the use of your data in this project, please contact the project maintainer.

## Current Dataset Summary

**Total Training Samples: 9,503** (consolidated dataset structure)

| Directory | Files | Label | Source | Description |
|-----------|-------|-------|--------|-------------|
| cry_baby | 4,552 | cry | Multiple sources (consolidated) | All baby cry recordings from Donate-a-Cry, ICSD, CryCeleb, HF, Kaggle |
| hard_negatives | 4,951 | non_cry | Multiple sources (consolidated) | All non-cry sounds: baby sounds, adult speech, environmental |
| noise | 1,960 | (augmentation only) | ESC-50 | Background noise for augmentation |
| cry | (archived) | N/A | Donate-a-Cry + HF + Kaggle | Legacy: Original cry samples |
| cry_ICSD | (archived) | N/A | ICSD real | Legacy: ICSD cry samples |
| cry_crycaleb | (archived) | N/A | CryCeleb2023 | Legacy: CryCeleb cry samples |
| baby_noncry | (archived) | N/A | Freesound.org + giulbia | Legacy: Baby non-cry sounds |
| adult_speech | (archived) | N/A | LibriSpeech | Legacy: Adult speech samples |
| environmental | (archived) | N/A | ESC-50 | Legacy: Environmental sounds |

**Class Balance: 4,552 cry / 4,951 non-cry (0.919:1 ratio - well-balanced)**

**Dataset Evolution:**
- **Original**: 696 cry samples (1:9.77 ratio)
- **October 2025 Additions**:
  - Added 2,312 real cries from ICSD dataset (424 strong + 1,888 weak labels)
  - Filtered out 883 snoring files and 500 synthetic files from ICSD
  - Added 69 unique cries from Hugging Face (after removing 1,128 duplicates/mislabeled)
  - Added 57 unique cries from Kaggle (after removing 997 duplicates)
  - Added 1,423 cries from CryCeleb2023 dataset
  - Total cry samples increased: 696 -> 4,557 (+554%)
  - Reduced adult_speech: 4,481 -> 2,241 (50% reduction for better balance)
  - Added additional environmental sounds: 1,960 -> 2,110
  - Class imbalance improved: 1:9.77 -> 0.968:1 (well-balanced)
  - Added .3gp and .mp3 format support
  - All cry samples are real recordings (no synthetic audio)
- **November 2025 Consolidation**:
  - Created cry_baby directory (4,552 files) consolidating all cry sources
  - Created hard_negatives directory (4,951 files) consolidating all non-cry sources
  - Total dataset: 9,503 samples (4,552 cry + 4,951 non-cry)
  - Class balance: 0.919:1 (cry:non-cry) - well-balanced
  - Legacy directories preserved for reference

**Consolidation History:**
- Merged baby_laugh and silence into baby_noncry (356 total)
- Removed duplicate ESC-50 files (noncry/household were duplicates)
- Renamed noise_speech to adult_speech for clarity
- Organized by sound type rather than source for better model logic
- Consolidated cry_1 and cry_2 temporary folders into main cry folder
- Created cry_ICSD with real recordings (strong + weak labels)
- Filtered ICSD: removed snoring (883) and synthetic (500) files for acoustic quality
- Archived original ICSD_cry folder structure
- Added cry_crycaleb with CryCeleb2023 dataset (1,423 files)
- **November 2025**: Consolidated all cry samples into cry_baby, all non-cry into hard_negatives

Last Updated: 2025-11-29
