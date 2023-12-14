import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import math
from xml.dom.minidom import parse


WFV = {'WFV1_2013':[5.8510, 7.1530, 8.3680, 7.4740],
       'WFV2_2013':[6.0140, 6.8230, 9.4510, 8.9960],
       'WFV3_2013':[5.8200, 6.2390, 7.0100, 7.7110],
       'WFV4_2013':[5.3500, 6.2350, 6.9920, 7.4620],
       'WFV1_2014':[0.2004, 0.1648, 0.1243, 0.1563],
       'WFV2_2014':[0.1733, 0.1383, 0.1122, 0.1391],
       'WFV3_2014':[0.1745, 0.1514, 0.1257, 0.1462],
       'WFV4_2014':[0.1713, 0.1600, 0.1497, 0.1435],
       'WFV1_2015':[0.1816, 0.1560, 0.1412, 0.1368],
       'WFV2_2015':[0.1684, 0.1527, 0.1373, 0.1263],
       'WFV3_2015':[0.1770, 0.1589, 0.1385, 0.1344],
       'WFV4_2015':[0.1886, 0.1645, 0.1467, 0.1378],
       'WFV1_2016':[0.1843, 0.1477, 0.1220, 0.1365],
       'WFV2_2016':[0.1929, 0.1540, 0.1349, 0.1359],
       'WFV3_2016':[0.1753, 0.1565, 0.1480, 0.1322],
       'WFV4_2016':[0.1973, 0.1714, 0.1500, 0.1572]}


OFFSET = {'WFV1_2013':[0.0039, 0.0047, 0.0030, 0.0274],
          'WFV2_2013':[0.0125, 0.0193, 0.0429, 0.0011],
          'WFV3_2013':[0.0071, 0.0334, 0.0226, 0.0117],
          'WFV4_2013':[0.0369, 0.0235, 0.0217, 0.0050]}


ESUN = {'WFV1': [1968.63, 1849.19, 1571.46, 1079.00],
        'WFV2': [1955.11, 1847.22, 1569.45, 1087.87],
        'WFV3': [1956.62, 1840.46, 1541.45, 1084.06],
        'WFV4': [1968.08, 1841.53, 1540.80, 1069.60]}



tiff_path = './GF1_datasets/bigpatch/bigpatch_tiff/'
txt_path = '/home/FAKEDATA/GF1_datasets/bigpatch/txt/bigpatch_index.txt'
xml_path = '/home/visint-book/liuyang_backup/Dataset/GF1/GF1_WFV_dataset/xml/'
save_path = '/home/FAKEDATA/GF1_datasets/bigpatch/bigpatch_tiff_eqTOA/'
os.makedirs(save_path, exist_ok=True)


for file in os.listdir(xml_path):
    content = file.replace('.xml','').split('_')

    day = content[-2]+'_'+content[-1]
    day_folder = tiff_path+day

    for patch in os.listdir(day_folder):
        print('--------------' + patch + '----------------')

        year = patch[:4]
        wfv_index = content[1]+'_'+year
        ESUN_index = content[1]

        DN = np.load(day_folder+'/'+patch).transpose(2, 0, 1)

        meta = parse(xml_path+file)
        root = meta.documentElement
        print(root.nodeName)
        SolarZenith = root.getElementsByTagName("SolarZenith")[0].firstChild.data
        angle = 90.00 - np.float(SolarZenith)
        radian = math.sin(math.pi * angle / 180)

        gain = np.expand_dims(WFV[wfv_index], axis=(1,2))

        if year == '2013':
            offset = np.expand_dims(OFFSET[wfv_index], axis=(1,2))
            L_lamda = (DN-offset)/gain
        else:
            L_lamda = gain * DN

        fenmu = np.asarray(ESUN[ESUN_index]) * math.sin(radian)
        fenmu = np.expand_dims(fenmu, axis=(1, 2))
        TOA = (math.pi * L_lamda) / fenmu

        TOA = TOA.transpose(1, 2, 0)

        np.save(save_path+patch,TOA)

