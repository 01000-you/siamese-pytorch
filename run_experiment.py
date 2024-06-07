import os
import shutil
import subprocess
from datetime import datetime
import argparse
from distutils.dir_util import copy_tree

# 실험 실행을 위한 인자 파서 설정
parser = argparse.ArgumentParser(description="Run multiple experiments")
parser.add_argument('--lrs', nargs='+', type=float, required=True, help='List of learning rates for the experiments')
parser.add_argument('--dr', nargs='+', type=float, required=True, help='List of dropout rates for the experiments')
parser.add_argument('--batch_sizes', nargs='+', type=int, required=True, help='List of batch sizes for the experiments')
args = parser.parse_args()

# Git 저장소의 경로
git_repo_path = "release"

for lr in args.lrs:
    for dr in args.drs:
        for batch_size in args.batch_sizes:
            # 현재 시간 기반으로 실험 이름 생성
            experiment_name = f"exp_lr_{lr}_dr_{dr}_bs_{batch_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # 실험 디렉터리 생성
            experiment_dir = os.path.join("experiments", experiment_name)
            os.makedirs(experiment_dir, exist_ok=True)

            # Git 저장소에서 코드를 복사
            shutil.copytree(git_repo_path, experiment_dir, dirs_exist_ok=True, symlinks=True)

            # 실험 디렉터리로 이동
            os.chdir(experiment_dir)

            # train.py 실행
            subprocess.run(["python", "train.py",
                            f"--exp_name={experiment_name}",
                            f"--lr={lr}",
                            f"--dr={dr}",
                            f"--batch_size={batch_size}",
                            f"--device=mps"])

            # 작업 디렉터리를 원래 위치로 변경
            os.chdir("../../../")
