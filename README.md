# Basic image processing to track _Cuscuta_'s coiling process

## General information

### Author

- **Erik J. Amézquita**, _University of Missouri_

### To whom correspondence should be addressed:

**Erik J. Amézquita**
240a Bond Life Sciences Center Building
Columbia, MO 65211
USA
eah4d***@***missouri.edu

### Date and geographic location of data collection

Data collected in Summer 2022 at the University of Missouri, Columbia. 

### Keywords

- Parasitic plants
- _Cuscuta campestris_
- Plant circumnutation
- Time-lapse photography

### Related content

- M. Bentelspacher, E.J. Amézquita, S. Adhikari, J. Barros, S.Y. Park (2024) The Early Dodder Gets the Host: Decoding the Coiling Patterns of Cuscuta campestris with Automated Image Processing. _Submitted_. Preprint DOI: [10.1101/2024.02.29.582789](https://doi.org/10.1101/2024.02.29.582789)


### License

MIT License

Copyright (c) 2024 Erik Amézquita

See `LICENSE` for additional details

### Acknowledgements

This research received support from the USDA-AFRI grant number 2023-67013-39896 awarded to SP, as well as funding from the Research Council, the College of Agriculture, Food, and Natural Resources (CAFNR), and the Interdisciplinary Plant Group (IPG) at the University of Missouri.

=========

## Data and file overview

### Overview

*Cuscuta* was grown in a greenhouse by infecting beets. Mature stems were then cut after 3 weeks and they were attached to bamboo skewers at different times of the day &mdash;9AM, 12PM, and 4PM. A camera was place in front of the _Cuscuta_-skewer setup. A snapshot was taken every 96 seconds for 24 hours, producing 900 images in total per repetition. There were 5 skewers per setup and 7 repetitions per inoculation time of the day, producing data for 105 individual _Cuscuta_ stems. Manual and automated observations were compared for Coiling success rates, and Initiation and completion times. Read Bentelspacher _et al._ (2024) for more context.

<video style="width:90%; margin: 0 auto; display: block;" controls>
	<source src="https://github.com/ejamezquita/ejamezquita.github.io/raw/main/cuscuta/video/9am_Inc_Rep_3_redone.mp4" type="video/mp4"></source>
</video>

These snapshots can be put together in a video like the example above. More videos can be seen in our [YouTube list](https://www.youtube.com/playlist?list=PLZkYcVyQr2u4tT0yoZAkrMqzQxRIDvxru).

### Tinkering with image processing

We developed a very ad-hoc pipeline with basic thresholding, erosion, and dilation operations. The main idea was to exploit color contrast between skewer and Cuscuta. This is by no means perfect, but allowed us to at least keep track of when Cuscuta crosses in front of the skewer, and when its angle and position stop changing. We were also able to treat Coil 1 and Coil 2 separately. 

### File description

The main folder is `jupyter`, which contains the Jupyter Python-based notebooks used to analyze the images. We used basic image processing libraries that should be easily available with any standard Python environment. Packages outside Python core used:

- `numpy`
- `scipy`
- `matplotlib`

The notebooks are numbered indicating which step of the analysis pipeline they cover.

Please read the notebooks themselves for more information.
