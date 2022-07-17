# Usage : python train_parallel.py --parallel True/False

from multiprocessing import Process
from functools import partial
import subprocess

from argparse import ArgumentParser

from src.utils import bool_flag

SCRIPT_PATH="./train.sh"

result = subprocess.run('chmod +x train.sh', shell=True, capture_output=True, text=True)
print(result)

def run_train(
    train_pct, weight_decay, representation_lr, decoder_lr, representation_dropout, decoder_dropout, opt, random_seed,
    operator, modular, p, task
):
    group_name=f"tdf={train_pct}-wd={weight_decay}-r_lr={representation_lr}-d_lr={decoder_lr}-r_d={representation_dropout}-d_d={decoder_dropout}-opt={opt}"
    print("Start Group name %s"%group_name)
    print(f"Random seed : {random_seed}")

    command=f"{SCRIPT_PATH} {train_pct} {weight_decay} {representation_lr} {decoder_lr} {representation_dropout} {decoder_dropout} {opt} {random_seed} {operator} {modular} {p} {task}"
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    stdoutdata, _ = process.communicate()

    if process.returncode != 0 :
        print("Error %s"%group_name)
    else :
        print("Success %s"%group_name)

    print("Finish Group name %s"%group_name)

    output = stdoutdata.decode("utf-8")
    print("*"*10)
    print(output)
    print("*"*10,"\n")

    #return stdoutdata

if __name__ == '__main__':
    parser = ArgumentParser(description="Grokking for MLP")
    parser.add_argument("--parallel", type=bool_flag, default=False)
    parallel = parser.parse_args().parallel

    operator="+"
    modular=False
    p=100
    task="classification"
    #task="regression"

    all_process = []
    for train_pct in [80] :
        for weight_decay in [0.0] :
            for representation_lr, decoder_lr in zip([0.001], [0.001]) : 
                for representation_dropout, decoder_dropout in zip([0.0], [0.0]) : 
                    for opt in ["adam"] :
                        for random_seed in [0, 100] :
                            if not parallel : 
                                run_train(
                                    train_pct, weight_decay, representation_lr, decoder_lr, representation_dropout, decoder_dropout, opt, random_seed,
                                    operator, modular, p, task
                                )
                            else :
                                task = partial(
                                    run_train, 
                                    train_pct, weight_decay, representation_lr, decoder_lr, representation_dropout, decoder_dropout, opt, random_seed,
                                    operator, modular, p, task
                                )
                                p = Process(target=task)
                                p.start()
                                all_process.append(p)
            
    for p in all_process : p.join()