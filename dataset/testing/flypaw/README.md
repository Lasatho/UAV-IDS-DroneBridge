# FlyPaw: Optimized route planning for scientific UAV missions

[https://doi.org/10.5061/dryad.0gb5mkm8d](https://doi.org/10.5061/dryad.0gb5mkm8d)

## Data Collection

This data was obtained in collaboration with North Carolina State University and the NSF Funded AERPAW Testbed-- Award #1939334

AERPAW (Aerial Experimentation and Research Platform for Advanced Wireless) is funded by the PAWR (Platforms for Advanced Wireless Research) Project Office. The authors thank the NC State team and the National Science Foundation for making these valuable resources available. More information about the AERPAW testbed can be found [here](https://aerpaw.org/)

## Description of the data and file structure

Several types of data are presented here, collected with the NSF-funded AERPAW testbed at North Carolina State University

1\) iperf measurements made from drone to base station using the srsRAN software-defined radio suite at various points around the flying field

2\) telemetry data from the drone

3\) srsRAN logs describing raw radio performance

4\) state information showing the transition of flight through various phases

Full description of the data and experiments best described in the paper below, with images: [https://doi.org/10.5061/dryad.0gb5mkm8d](https://doi.org/10.5061/dryad.0gb5mkm8d)

Code and results are also available in the GitHub repository:
[https://github.com/FlyNet-NSF/flypaw](https://github.com/FlyNet-NSF/flypaw)

## Data Format

Data is shared as JSON from iperf results, from drone telemetry results, and from dynamic planning algorithms described in the paper.

Radio raw results shared from [srsRAN ](https://docs.srsran.com/projects/4g/en/latest/usermanuals/source/srsenb/source/2_enb_getstarted.html#observing-results)and described in the link.

### Files and variables

#### File: flypawState\_20220311-120218.json

**Description:** 

Detailed state information as the automated flight transitions

#### File: 2022-01-13\_135723\_radio\_log.txt

#### File: 2022-01-13\_135715\_radio\_epc\_log.txt

#### File: 2022-01-13\_135715\_radio\_enb\_log.txt

#### File: 2022-03-11\_12\_02\_18\_radio\_log.txt

#### File: 2022-03-11\_12\_01\_50\_radio\_epc\_log.txt

#### File: 2022-03-11\_12\_01\_50\_radio\_enb\_log.txt

**Description:** 

srsRAN logs for the SDN suite

#### File: iperf3\_20220113-135727.json

#### File: iperf3\_20220311-120218.json

#### File: 2022-01-13\_135715\_iperfserver\_log.txt

#### File: 2022-03-11\_12\_01\_50\_iperfserver\_log.txt

**Description:** 

TCP and UDP-based iperf data describing air-to-ground UDP throughput at various points in the test area as estimated by the vehicle and also at the ground

#### File: 2022-01-13\_135723\_vehicle\_log.txt

#### File: 2022-01-13\_135723\_vehicleOut.txt

**Description:** 

Default AERPAW vehicle logs

#### File: telemetry\_20220311-120218.json

**Description:** 

Telemetry from the vehicle during the flight
