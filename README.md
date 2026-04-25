## 1. Code sturcture
The implement of MAPPO and IPPO located in onpolicy folder, and the implement of QMIX is located in the offpolicy folder.
Inside the folder, it included both the implement and the cooperative multi-agent environment.


## 2. Installation
Install the following packages

``` Bash
# create conda environment
conda create -n marl python==3.8
conda activate marl
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install wandb
git clone https://github.com/Chen4513/Reimplement-of-MAPPO-in-multi-agent-benchmark.git
cd Reimplement-of-MAPPO-in-multi-agent-benchmark
pip install -r requirements.txt
```

### 2.1 StarCraftII [4.10](http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip)

``` Bash
unzip SC2.4.10.zip
# password is iagreetotheeula
echo "export SC2PATH=~/StarCraftII/" > ~/.bashrc
```

* download SMAC Maps, and move it to `~/StarCraftII/Maps/`.

* To use a stableid, copy `stableid.json` from https://github.com/Blizzard/s2client-proto.git to `~/StarCraftII/`.

### 2.2 MPE

``` Bash
# install this package first
pip install seaborn
```

There are 2 Cooperative scenarios in MPE:

* simple_spread
* simple_reference

## 3.Train
Here we train mpe environment as an example:
```
cd onpolicy/scripts/train_mpe_scripts
export WANDB_MODE=offline
chmod +x ./train_mpe_spread.sh
./train_mpe_spread.sh
```
Local results are stored in subfold scripts/results. Note that we use Weights & Bias as the default visualization platform; to use Weights & Bias, please register and login to the platform first. More instructions for using Weights&Bias can be found in the official [documentation](https://docs.wandb.ai/). Adding the `--use_wandb` in command line or in the .sh file will use Tensorboard instead of Weights & Biases. 

## 4. Publication

The original paper could be found in [paper](https://arxiv.org/abs/2103.01955):
```
@misc{yu2021surprising,
      title={The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games}, 
      author={Chao Yu and Akash Velu and Eugene Vinitsky and Jiaxuan Gao and Yu Wang and Alexandre Bayen and Yi Wu},
      year={2021},
      eprint={2103.01955},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

