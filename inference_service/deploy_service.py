ACTIVE = {
    "job_id": None,
    "task": None,
}


def deploy(job_id: str, task: str):
    ACTIVE["job_id"] = job_id
    ACTIVE["task"] = task
    return ACTIVE


def get_status():
    return ACTIVE