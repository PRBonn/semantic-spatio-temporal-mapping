<div align="center">
  <h1>Semantic Spatio-Temporal Mapping</h1>
  <h1>REPRODUCIBILITY BRANCH</h1>
    <a href="https://github.com/PRBonn/semantic-spatio-temporal-mapping#Installation"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
    <a href="https://github.com/PRBonn/semantic-spatio-temporal-mapping#Usage"><img src="https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54" /></a>
    <a href="https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/lobefaro2025ral.pdf"><img src="https://img.shields.io/badge/Paper-pdf-<COLOR>.svg?style=flat-square" /></a>
    <a href="https://github.com/PRBonn/semantic-spatio-temporal-mapping/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" /></a>

<p>
  <img src="https://github.com/PRBonn/semantic-spatio-temporal-mapping/blob/main/images/first_image.png" width="400"/>
</p>

<p>
  <i>Semantic Spatio-Temporal Mapping is a RGB-D odometry and mapping system capable to produce maps with spatio-temporal consistent fruit instance segmentation.</i>
</p>

</div>


## Introduction 
You are in the reproducibility branch. Note: this branch is here for the only purpose of allowing reproducibility for our [paper](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/lobefaro2025ral.pdf).
If you are not here for this reason, go back to the main branch. Otherwise, follow the instructions.


# Dependencies
You need to install docker and docker-compose. Please, search by yourself how to install them.
You need a GPU with CUDA support also.


# Usage
In order to run the code and obtain the same numbers showed in the paper, you first need to get the data. For that, please [contact us](mailto:llobefar@uni-bonn.de).

Once you have the data, open the file ```docker-compose.yml``` and substitute "PATHTODATASET" with the path to the dataset on your machine.

Finally, you can just run the following commands:

```
docker-compose build
docker-compose run --rm ral2025_reprod
```

It will build the docker image and then build and run the pipeline inside the container. You will see the outputs on screen, with the results at the end. Be patient while building.


## Citations and LICENSE
This project is free software made available under the MIT License. For details see the [LICENSE](https://github.com/PRBonn/semantic-spatio-temporal-mapping/blob/main/LICENSE) file.

If you use this project, please refer to our [paper on data association](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/lobefaro2023iros.pdf), [paper on plants deformation](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/lobefaro2024iros.pdf) and [paper on spation-temporal semantic mapping](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/lobefaro2025ral.pdf):

```bibtex
@inproceedings{lobefaro2023iros,
  author = {L. Lobefaro and M.V.R. Malladi and O. Vysotska and T. Guadagnino and C. Stachniss},
  title = {{Estimating 4D Data Associations Towards Spatial-Temporal Mapping of Growing Plants for Agricultural Robots}},
  booktitle = iros,
  year = 2023,
  codeurl = {https://github.com/PRBonn/plants_temporal_matcher}
}
```
```bibtex
@inproceedings{lobefaro2024iros,
  author = {L. Lobefaro and M.V.R. Malladi and T. Guadagnino and C. Stachniss},
  title = {{Spatio-Temporal Consistent Mapping of Growing Plants for Agricultural Robots in the Wild}},
  booktitle = iros,
  year = 2024,
  codeurl = {https://github.com/PRBonn/spatio_temporal_mapping}
}
```
```bibtex
@inproceedings{lobefaro2025ral,
  author = {L. Lobefaro and M. Sodano and D. Fusaro and F. Magistri and M.V.R. Malladi and T. Guadagnino and A. Pretto and C. Stachniss},
  title = {{Spatio-Temporal Consistent Semantic Mapping for Robotics Fruit Growth
Monitoring}},
  journal = ral,
  year = {2025},
  codeurl = {https://github.com/PRBonn/semantic-spatio_temporal_mapping},
  volume = {},
  number = {},
  pages = {},
  issn = {},
  doi = {},
}
```


## Acknowledgement
The code structure of this software follows the same of [KISS-ICP](https://github.com/PRBonn/kiss-icp) and some code is re-used from that repo. Please, if you use this software you should at least acknowledge also the work from KISS-ICP by giving a star on GitHub.


## Further Notes
This work is based on our [previous work](https://github.com/PRBonn/spatio-temporal-mapping) on spatio-temporal consistent mapping. They share the same codebase and this is considered an extension of our previous pipeline. Stay tuned for news.
