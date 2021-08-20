# DS4CG-Automatic-Georeferencing
To get a better understanding of this project before using it, please watch this short video: and/or read this short paper: . This project was created by Collin Giguere and Sowmya Vasuki Jallepalli for the Data Science for the Common Good program in Summer 2021 in partnership with UMass Libraries and Department of Environmental Conservation and advising from Pixel Forensics. To find out more about the program, see https://www.ds.cs.umass.edu/ds4cg.

## Setup
Before using the application, you must set up the environment. First install python on your device by following the steps here: https://www.python.org/downloads/. Make sure you download and install **Python 3.9**. Then install GIT from here: https://git-scm.com/downloads. If you need a basic introduction on how to use the command prompt on Windows, start here: https://www.bleepingcomputer.com/tutorials/windows-command-prompt-introduction/; and an introduction to command line on MacOS and Unix here: https://datacarpentry.org/shell-genomics/01-introduction/.
The bare minumum understanding to use this application is a basic understanding of what the command line is (basically a file explorer) and the commands 'ls' (MacOS/Unix), 'dir' (same command on Windows), and 'cd' (both Windows and MacOS/Unix). If you understand these, then you can move forward.

Next, open the Terminal (on MacOS) or Command Prompt (on Windows), navigate to the directory you want the application in, and execute the following commands. You can copy and paste them one by one or all at once.

### Windows
`git clone https://github.com/cdgiguere/DS4CG-Automatic-Georeferencing.git`

`cd DS4CG-Automatic-Georeferencing\`

`py -m pip install --upgrade pip`

`py -m pip install --user virtualenv`

`py -m venv env`

`.\env\Scripts\activate`

##### On 32-bit systems:
`py -m pip install -r Setup\requirements_32.txt`

##### On 64-bit systems:
`py -m pip install -r Setup\requirements_64.txt`

### MacOS/UNIX
`git clone https://github.com/cdgiguere/DS4CG-Automatic-Georeferencing.git`

`cd DS4CG-Automatic-Georeferencing/`

`python3 -m pip install --user --upgrade pip`

`python3 -m pip install --user virtualenv`

`python3 -m venv env`

`source env/bin/activate`

`python3 -m pip install -r Setup\requirements_mac.txt`

### Config
Included with each of the two pipelines is a config yaml file. These are used to specify where files are located, where to place output files, and certain other parameters to use. Instructions on how to fill out these files are in the files themselves; simply open them with any text editor. They are unique for the two pipelines; one in the Propagation directory and one in the Satellite directory.

## Use
This setup and congifuration only needs to be done one per device. All you need to do to run the application is make sure the environment is activated (it already will be if you just completed the setup, but you will need to activate it each new session) with the same command `.\env\Scripts\activate` (Windows) or `source env/bin/activate` (MacOS/Unix), and run these commands:

### Windows
`cd Propagation\`

`py Propagate.py cni3h80 cni3h81 cni3h82 ...`

or

`cd Satellite\`

`py Satellite.py cni3h79`

### MacOS/UNIX
`cd Propagation/`

`python3 Propagate.py cni3h80 cni3h81 cni3h82 ...`

or

`cd Satellite/`

`python3 Satellite.py cni3h81`

In these examples cni3h80 is treated as the referenced file and cni3h81 and cni3h82 are treated as the unreferenced files.

#### Note
If you are using this application for its orignial developed purpose: the MacConnell set; you can just use the image codes like above as long as the config file is defined correctly. If you are working on another dataset, first fill out the config accordingly and then you will specify the file paths such as:

`py Propagate.py path\to\referenced_image path\to\unreferenced_image path\to\unreferenced_image ...`

Note: The Satellite pipeline is not currently available for anything but the MacConnell set since more information is needed about the location and size of the unreferenced image.
