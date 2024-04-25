conda create -n grounded_decoding_cu11 python=3.9
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge accelerate
pip install -e ./transformers
pip install datasets evaluate

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/ameya/mambaforge/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/ameya/mambaforge/etc/profile.d/conda.sh" ]; then
        . "/home/ameya/mambaforge/etc/profile.d/conda.sh"
    else
        export PATH="/home/ameya/mambaforge/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<