import os
import torch
import timm.models.hub as timm_hub

def download_cached_file(url, check_hash=True, progress=False):
    """
    Download a file from a URL and cache it locally. If the file already exists, it is not downloaded again.
    If distributed, only the main process downloads the file, and the other processes wait for the file to be downloaded.
    """

    def get_cached_file_path():
        # a hack to sync the file path across processes
        parts = torch.hub.urlparse(url)
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(timm_hub.get_cache_dir(), filename)

        return cached_file

    
    timm_hub.download_cached_file(url, check_hash, progress)

    # if is_dist_avail_and_initialized():
    #     dist.barrier()

    return get_cached_file_path()


url = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth"
cached_file = download_cached_file(
        url, check_hash=False, progress=True
    )

# Same issue: https://github.com/salesforce/LAVIS/issues/210