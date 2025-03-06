# Video Sub Extractor

![python version](https://img.shields.io/badge/Python-3.12-blue)
![support os](https://img.shields.io/badge/OS-Windows-green.svg)

A free program that extracts hard coded subtitles from a video and generates an external subtitle file.

<img src="docs/images/gui%20screenshot.png" width="500">


**Features**

- Detect subtitle area by searching common area.
- Manual resize or change of subtitle area (click and drag mouse to perform).
- Single and Batch subtitle detection and extraction.
- Start and Stop subtitle extraction positions can be selected (use arrow keys for precise selection).
- Resize video display (Zoom In (Ctrl+Plus), Zoom Out (Ctrl+Minus)).
- Non subtitle area of the video can be hidden to limit spoilers.
- Toast Notification available on Windows upon completion of subtitle detection and extraction.
- [Preferences docs](docs/Preferences.md) available for modification of options when extraction subtitles.
- Multiple languages supported through [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR). They will be
  automatically downloaded as needed.

Generated subtitles can be translated with this [script](https://github.com/voun7/Subtitle_Translator).

**Supported languages**

| Language            | Abbreviation | | Language          | Abbreviation | | Language     | Abbreviation |
|---------------------|--------------|-|-------------------|--------------|-|--------------|--------------|
| Chinese & English   | ch           | | Arabic            | ar           | | Avar         | ava          |
| English             | en           | | Hindi             | hi           | | Dargwa       | dar          |
| French              | fr           | | Uyghur            | ug           | | Ingush       | inh          |
| German              | german       | | Persian           | fa           | | Lak          | lbe          |
| Japanese            | japan        | | Urdu              | ur           | | Lezghian     | lez          |
| Korean              | korean       | | Serbian(latin)    | rs_latin     | | Tabassaran   | tab          |
| Chinese Traditional | chinese_cht  | | Occitan           | oc           | | Bihari       | bh           |
| Italian             | it           | | Marathi           | mr           | | Maithili     | mai          |
| Spanish             | es           | | Nepali            | ne           | | Angika       | ang          |
| Portuguese          | pt           | | Serbian(cyrillic) | rs_cyrillic  | | Bhojpuri     | bho          |
| Russian             | ru           | | Bulgarian         | bg           | | Magahi       | mah          |
| Ukranian            | uk           | | Estonian          | et           | | Nagpur       | sck          |
| Belarusian          | be           | | Irish             | ga           | | Newari       | new          |
| Telugu              | te           | | Croatian          | hr           | | Goan Konkani | gom          |
| Sanskrit            | sa           | | Hungarian         | hu           | | Norwegian    | no           |
| Tamil               | ta           | | Indonesian        | id           | | Polish       | pl           |
| Afrikaans           | af           | | Icelandic         | is           | | Romanian     | ro           |
| Azerbaijani         | az           | | Kurdish           | ku           | | Slovak       | sk           |
| Bosnian             | bs           | | Lithuanian        | lt           | | Slovenian    | sl           |
| Czech               | cs           | | Latvian           | lv           | | Albanian     | sq           |
| Welsh               | cy           | | Maori             | mi           | | Swedish      | sv           |
| Danish              | da           | | Malay             | ms           | | Swahili      | sw           |
| Maltese             | mt           | | Adyghe            | ady          | | Tagalog      | tl           |
| Dutch               | nl           | | Kabardian         | kbd          | | Turkish      | tr           |
| Uzbek               | uz           | | Vietnamese        | vi           | | Mongolian    | mn           | 
| Abaza               | abq          |

**Download**

[Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist) must be
installed. The program will not start without it.

- [Windows CPU Version](https://github.com/voun7/Video_Sub_Extractor/releases/download/v1.0/VSE-windows-cpu.zip)

## Demo Video

[![Demo Video](docs/images/demo%20screenshot.png)](https://youtu.be/nnm_waobgnI "Demo Video")

## Setup Instructions

### Download and Install:

[Latest Version of Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist)

Install packages

```commandline
pip install -r requirements.txt
```

Run `gui.py` to use Graphical interface and `main.py` to use Terminal.

### Compile Instructions

Run `compiler.py` to build compiled program

