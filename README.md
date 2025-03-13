# Voice-Guidance-Parking-Finder

주차장의 남은공간을 detecting 하여 주차장의 남은공간을 음성으로 알려주는 역할.

## Requirement
* 10th generation Intel® CoreTM processor onwards
* At least 32GB RAM
* Python 3.9.13
  
## Clone code

* (Code clone 방법에 대해서 기술)

```shell
git clone https://github.com/97JongYunLee/Voice-Guidance-Parking-Finder.git
```

## Prerequite

* (프로젝트를 실행하기 위해 필요한 dependencies 및 configuration들이 있다면, 설치 및 설정방법에 대해 기술)

```shell
python -m venv .venv
.venv/Scripts/activate

python -m pip install -U pip
python -m pip install wheel

python -m pip install openvino-dev
git clone --recurse-submodules https://github.com/openvinotoolkit/open_model_zoo.git

cd open_model_zoo
python -m pip install -r requirements.txt

cd open_model_zoo/demos/text_to_speech_demo/python
omz_downloader --name text-to-speech-en-0001-duration-prediction
omz_downloader --name text-to-speech-en-0001-generation
omz_downloader --name text-to-speech-en-0001-regression
omz_converter --name text-to-speech-en-0001-duration-prediction
omz_converter --name text-to-speech-en-0001-generation
omz_converter --name text-to-speech-en-0001-regression

cd open_model_zoo/demos/object_detection_demo/python
omz_downloader --name vehicle-detection-0202
omz_converter --name vehicle-detection-0202


## Steps to build

* (프로젝트를 실행을 위해 빌드 절차 기술)



## Steps to run

* (프로젝트 실행방법에 대해서 기술, 특별한 사용방법이 있다면 같이 기술)

```shell 
.venv/bin/activate

cd open_model_zoo/demos/text_to_speech_demo/python
python parking.py
```

## Output

![test_0706_reuslt](https://github.com/97JongYunLee/Voice-Guidance-Parking-Finder/assets/139088562/37b53368-0709-4a63-b311-3b4c309f3db8)




## Appendix

* 본 프로젝트는 첨부된 이미지파일로만 사용가능.
* 다른 이미지파일로 실행하려면 코드수정 필요
