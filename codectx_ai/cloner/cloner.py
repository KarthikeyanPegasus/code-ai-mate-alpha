import subprocess


def clone_github_repository(url, local_path):
    try:
        subprocess.run(['git', 'clone', url, local_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone cloner: {e}")
        return False


def prepare_repository(list_path: str, repo_path: str):
    with open(list_path) as f:
        for line in f:
            url = line.strip()
            repo_name = url.split("/")[-1]
            local_path = f"{repo_path}/{repo_name}"
            print(f"Cloning repository: {url}")
            if clone_github_repository(url, local_path):
                print(f"Repository cloned successfully at {local_path}")
            else:
                print(f"Failed to clone the cloner: {url}")
