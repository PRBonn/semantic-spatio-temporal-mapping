# MIT License
#
# Copyright (c) 2024 Luca Lobefaro, Meher V.R. Malladi, Tiziano Guadagnino, Cyrill Stachniss
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from pathlib import Path
from typing import Optional
from st_mapping.datasets import (
    available_dataloaders,
    dataloader_name_callback,
    dataset_factory,
)

import typer

from st_mapping.config.parser import load_config

from st_mapping.semantic_mapping_onref_pipeline import SemanticMappingOnRefPipeline


app = typer.Typer(no_args_is_help=True, add_completion=False, rich_markup_mode="rich")



@app.command()
def st_mapping_semantic_mapping_onref(
    dataset_folder: Path = typer.Argument(
        ..., help="Path to the dataset folder.", show_default=False
    ),
    reference_dataset_folder: Path = typer.Argument(
        ..., help="PAth to the reference dataset folder.", show_default=False
    ),
    dataloader: str = typer.Option(
        None,
        show_default=False,
        case_sensitive=False,
        autocompletion=available_dataloaders,
        callback=dataloader_name_callback,
        help="[Optional] Use a specific dataloader from those supported.",
    ),
    config_filename: Optional[Path] = typer.Option(
        None,
        "--config",
        exists=True,
        show_default=False,
        help="[Optional] Path to the configuration file",
    ),
    visualize: bool = typer.Option(
        False,
        "--visualize",
        "-v",
        help="[Optional] Open an online visualization of the mapping system",
        rich_help_panel="Additional Options",
    ),
    degradation_depth_level: int = typer.Option(
        0,
        "--degradation_depth_level",
        "-d",
        help="[Optional] Set the degradation level to test robustness on depth noise. Accepted values [0 (default), 1, 2, 3]",
        rich_help_panel="Additional Options",
    ),
):
    # Argument parsing
    if not dataloader:
        dataloader = "generic"
    config = load_config(config_filename)
    if not degradation_depth_level in [0, 1, 2, 3]:
        print(
            "[ERROR] Degradation depth level can be only one of these values [0, 1, 2, 3]"
        )
        exit(1)

    # Initialization
    dataset = dataset_factory(
        dataloader=dataloader,
        dataset_folder=dataset_folder,
        has_poses=False,
        depth_scale=config.dataset.depth_scale,
        degradation_depth_level=degradation_depth_level,
    )
    ref_dataset = dataset_factory(
        dataloader=dataloader,
        dataset_folder=reference_dataset_folder,
        has_poses=True,
        depth_scale=config.dataset.depth_scale,
        degradation_depth_level=degradation_depth_level,
    )

    # Run pipeline
    SemanticMappingOnRefPipeline(
        dataset=dataset,
        ref_dataset=ref_dataset,
        config=config,
        visualize=visualize,
    ).run()


def run():
    app()
