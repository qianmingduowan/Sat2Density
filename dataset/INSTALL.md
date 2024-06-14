For reproduce our paper,

you should first download 4 zip file:

`
CVACT/satview_correct.zip , 
CVACT/streetview.zip , 
CVUSA/bingmap/19.zip ,
CVUSA/streetview/panos.zip
`
 from [here](https://anu365-my.sharepoint.com/:f:/g/personal/u6293587_anu_edu_au/EuOBUDUQNClJvCpQ8bD1hnoBjdRBWxsHOVp946YVahiMGg?e=F4yRAC), the project page is [Sat2StrPanoramaSynthesis](https://github.com/shiyujiao/Sat2StrPanoramaSynthesis).

Then download the sky mask and data split from [here](https://drive.google.com/drive/folders/1pfzwONg4P-Mzvxvzb2HoCpuZFynElPCk?usp=sharing)

Last，the users should organize the dataset just like:
```
├dataset
├── CVACT
│   ├── streetview
│   ├── satview_correct
│   ├── pano_sky_mask
│   ├── ACT_data.mat
└── CVUSA
│   ├── bingmap
│   │   ├── 19
│   └── streetview
│   │   ├── panos
│   ├── sky_mask
│   ├── splits
```

Tip： The sky masks are processed with [Trans4PASS](https://github.com/jamycheung/Trans4PASS).
