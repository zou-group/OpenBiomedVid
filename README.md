<a name="readme-top"></a>

<div align="center">

# OpenBiomedVid: How Well Can General Vision-Language Models Learn Medicine By Watching Public Educational Videos?

</div>

## Introduction

Publicly available biomedical videos, such as those on YouTube, serve as valuable educational resources for medical students. Unlike standard machine learning datasets, these videos are designed for human learners, often mixing medical imagery with narration, explanatory diagrams, and contextual framing. In this work, we investigate whether such pedagogically rich, yet non-standardized and heterogeneous videos can effectively teach general-domain vision-language models biomedical knowledge. To this end, we introduce OpenBiomedVid, a biomedical video instruction tuning dataset comprising 1031 hours of video-caption and Q/A pairs, curated through a multi-step human-in-the-loop pipeline. Diverse biomedical video datasets are rare, and OpenBiomedVid fills an important gap by providing instruction-style supervision grounded in real-world educational content. Surprisingly, despite the informal and heterogeneous nature of these videos, the fine-tuned Qwen-2-VL models exhibit substantial performance improvements across most benchmarks. The 2B model achieves gains of 98.7\% on video tasks, 71.2\% on image tasks, and 0.2\% on text tasks. The 7B model shows improvements of 37.09\% on video and 11.2\% on image tasks, with a slight degradation of 2.7\% on text tasks compared to their respective base models. To address the lack of standardized biomedical video evaluation datasets, we also introduce two new expert curated benchmarks, MIMICEchoQA and SurgeryVideoQA. On these benchmarks, the 2B model achieves gains of 99.1\% and 98.1\%, while the 7B model shows gains of 22.5\% and 52.1\%, respectively, demonstrating the models' ability to generalize and perform biomedical video understanding on cleaner and more standardized datasets than those seen during training. These results suggest that educational videos created for human learning offer a surprisingly effective training signal for biomedical VLMs.


![pipeline](assets/figures/dataset_pipeline.png)

![pipeline-example](assets/figures/dataset_comparison.png)

### Main results
<a align="center">
    <img src="assets/figures/multimodal_benchmarks.png" width="100%">
    <!-- Text. -->
</a>


### Dataset Statistics
<a align="center">
    <img src="assets/figures/finetuning_dataset_statistics_hours.png" width="100%">
    <!-- Text. -->
</a>


## Installation

```bash
conda create -n openbiomedvid python=3.10
conda activate openbiomedvid
pip install -r requirements.txt
pip install -e .

# install flash-attn
pip install flash-attn --no-build-isolation
conda install -c conda-forge moviepy
```

⚠️ Note: If the above installation for flash-attn fails, please clone and build from source:

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install
```

Clone the Liger repository
```bash
git clone https://github.com/linkedin/Liger-Kernel.git
cd Liger-Kernel
pip install -e .
```

## 🤗 Datasets
- [OpenBiomedVid](https://huggingface.co/datasets/connectthapa84/OpenBiomedVid): Instruction tuning dataset
- [SurgeryVideoQA](https://huggingface.co/datasets/connectthapa84/SurgeryVideoQA): Benchmark for surgical video QA
- [MIMICEchoQA](https://huggingface.co/datasets/connectthapa84/MIMICEchoQA): Benchmark for echocardiogram QA (requires PhysioNet download)

```python
from datasets import load_dataset

# Load datasets
openbiomedvid = load_dataset("connectthapa84/OpenBiomedVid")
surgery_qa = load_dataset("connectthapa84/SurgeryVideoQA")
mimic_echo = load_dataset("connectthapa84/MIMICEchoQA")
```

**⚠️ Notes on Video Access**
- We do not provide raw YouTube videos due to copyright.
- You must download them separately for both OpenBiomedVid and SurgeryVideoQA.
- For MIMICEchoQA, download the official echo videos from PhysioNet:
🔗 https://physionet.org/content/mimic-iv-echo/0.1/


We provide our data curation pipeline in `src/dataset`. However, you do not need to rerun these scripts, as we provide you with huggingface dataset already. All you need to do is download the respective videos, and follow steps below to segment the video to pair with our huggingface dataset. 

### Slice Raw Videos into Segments

After downloading the raw videos into a directory (e.g., `videos/`), you can extract segments referenced in the dataset using:

```bash
python src/openbiomedvid/dataset/slice_videos.py \
  --dataset OpenBiomedVid \
  --input_dir /data/rahulthapa/OpenBiomedVid_test/videos \
  --output_dir /data/rahulthapa/OpenBiomedVid_test/video_segments \
  --num_processes 32
```

**This step is mandatory before training or evaluation.**

### Training

We provide our training package under `tore-train`. The training pipeline is currently set up for streaming data from S3, but can be modified to use local paths.

Scripts are numbered for clarity and include:
- Video preprocessing
- Dataset creation
- Model training

### 📊 Evaluation

Evaluation scripts are provided in `src/openbiomedvid/evaluation`.

To run a demo inference using `Qwen/Qwen2-VL-7B-Instruct` on the `SurgeryVideoQA` benchmark:

- Make sure you have already preprocessed the videos (see [Slice Raw Videos into Segments](#-slice-raw-videos-into-segments)).
- Then run the evaluation script as shown in the demo (`src/openbiomedvid/evaluation/1_inference.py`).
- Evaluate the model using GPT evaluator (`src/openbiomedvid/evaluation/3_gpt_eval.py`).


### Citation

```bibtex
Coming soon
```

### Our Team
<table>
	<tbody>
		<tr>
            <td align="center">
                <a href="https://www.linkedin.com/in/rahul-thapa/">
                    <img src="https://media.licdn.com/dms/image/v2/D5603AQFc9Bdg5VEPxQ/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1683671172066?e=1750291200&v=beta&t=ljktIYvNmUhAs5Deim2AeqwzxYM2unVVf9tUlZQxCKI" height="100;" alt="rthapa84"/>
                    <br />
                    <sub><b>Rahul Thapa</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://www.linkedin.com/in/andrewli2403/">
                    <img src="https://media.licdn.com/dms/image/v2/D5603AQGWbwUI-Jp5UQ/profile-displayphoto-shrink_800_800/B56ZPa.T1pGQAg-/0/1734545587833?e=1750291200&v=beta&t=S11mHJg-t_e0bnzMNb65CwIXR92WmAz4DAm2N7p877w" height="100;" alt="andrewli"/>
                    <br />
                    <sub><b>Andrew Li</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://www.linkedin.com/in/qingyang-wu-2497a0110/">
                    <img src="https://media.licdn.com/dms/image/v2/D5603AQG2ABA6rFYnyA/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1714876212778?e=1750291200&v=beta&t=2iAgZVm03rcSPpJShK2JGeKIewVFU8ufvFrsFxwvteI" height="100;" alt="qingyangwu"/>
                    <br />
                    <sub><b>Qingyang Wu</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://ai.stanford.edu/~bryanhe/">
                    <img src="https://ai.stanford.edu/~bryanhe/img/photo.jpg?1" height="100;" alt="bryanhe"/>
                    <br />
                    <sub><b>Bryan He</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://x.com/Yuki_Sahashi">
                    <img src="https://media.licdn.com/dms/image/v2/D5603AQFQLz-mIt1_LA/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1728526899115?e=1750291200&v=beta&t=mopxxo3N-noOZZtTjR5-piVbLAN-sOqa6Ftm1vIF9mE" height="100;" alt="yukisahashi"/>
                    <br />
                    <sub><b>Yuki Sahashi</b></sub>
                </a>
            </td>
		</tr>
        <tr>
            <td align="center">
                <a href="https://www.linkedin.com/in/christina-binder-rodriguez-md-phd-690b6815b/">
                    <img src="https://media.licdn.com/dms/image/v2/D4D03AQGwPVnM3Bkxmg/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1727554556896?e=1750291200&v=beta&t=mPCZY4hV5C-_o6rPy8ZixIaxPYUYi4rsI8fOBI8Ba9U" height="100;" alt="christinabinder"/>
                    <br />
                    <sub><b>Christina Binder</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://scholar.google.com/citations?user=zafRY0sAAAAJ&hl=en">
                    <img src="https://scholar.googleusercontent.com/citations?view_op=medium_photo&user=zafRY0sAAAAJ&citpid=4" height="100;" alt="angelazhang"/>
                    <br />
                    <sub><b>Angela Zhang</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://x.com/ben_athi?lang=en">
                    <img src="https://pbs.twimg.com/profile_images/1585755523198210054/uSon6fvn_400x400.jpg" height="100;" alt="benathi"/>
                    <br />
                    <sub><b>Ben Athiwaratkun</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://shuaiwen-leon-song.github.io/">
                    <img src="https://shuaiwen-leon-song.github.io/images/Shuaiwen-Leon-Song.png" height="100;" alt="shuaiwensong"/>
                    <br />
                    <sub><b>Shuaiwen Leon Song</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://douyang.github.io/">
                    <img src="https://douyang.github.io/photo.jpg" height="100;" alt="douyang"/>
                    <br />
                    <sub><b>David Ouyang</b></sub>
                </a>
            </td>
        </tr>
        <tr>
            <td align="center">
                <a href="https://www.james-zou.com/">
                    <img src="https://static.wixstatic.com/media/0f3e8f_cfa7e327b97745ddb8c4a66454b5eb3e~mv2.jpg/v1/fill/w_398,h_557,al_c,q_80,usm_0.66_1.00_0.01,enc_avif,quality_auto/46824428A5822_ForWeb.jpg" height="100;" alt="jameszou"/>
                    <br />
                    <sub><b>James Zou</b></sub>
                </a>
            </td>
        </tr>
	<tbody>
</table>
