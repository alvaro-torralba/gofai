# Good Old-Fashioned AI

This repository contains the learning scripts for learning partial grounding and pruning rules models.

## Installation

Install  dependencies:

```
apt-get update
apt-get -y install --no-install-recommends cmake make g++ python3 automake pkg-config libgmp10 libgmp-dev libreadline-dev  libsqlite3-dev sqlite3 python3-pip python3-venv swig build-essential libpython3-dev
```
The dependency yap must be installed locally. To do so, follow the following instructions.

```
mkdir -p /learning/yap/build
cd /learning/yap/build
cmake ..
make
make install
```

Create virtual environment: 
```
python3 -m venv .venv
.venv/bin/pip install -r /requirements-learn.txt
```

Compile fd-symbolic and fd-partial-grounding (go to directory and run ./build.py)


## Usage

```
source .venv/bin/activate
```

# Learn from a set of instances: 

```
python3 learn.py <directory_with_set_of_instances> --domain_knowledge_file <name> --cpus 1 --total_time_limit 86400 --total_memory_limit 90000
```

# Learn from instances for which the good operators have already been computed

```
python3 learn-from-data.py <directory_with_training_data> 
```



# Dependencies

The following dependencies are included.

## Yap

Taken from: https://github.com/vscosta/yap

Yap is distributed under the   LGPL  licence terms. For details visit http://www.gnu.org/copyleft/lesser.html.

## Aleph

Taken from: https://www.cs.ox.ac.uk/activities/programinduction/Aleph/aleph.html

## Translate

Taken from:  https://www.fast-downward.org/  (release 22-12)

Fast Downward is licensed under the GNU Public License (GPL), as described in the main repository. If you want to use the planner in any way that is not compatible with the GPL (v3 or newer), you will have to get permission from us.
