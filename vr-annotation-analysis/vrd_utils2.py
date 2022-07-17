#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 10:25:30 2021

@author: dave
"""

'''
This module defines variables that support the analysis of the
VRD images and annotations. 
'''

#%% support for the analysis of object class 'glasses'

# Explanation:

# We defined 6 relationships found to commonly appear in VRD image annotations 
# when 'glasses' is used to refer to *eyeglasses*. These are:
#
# (person, wear, glasses), (glasses, on, person), 
# (person, has, glasses), (glasses, on, face),
# ('helmet', 'above', 'glasses'), ('person', 'in', 'glasses')
#
# We used these 6 relationships as *filtering conditions*.  Our 
# analysis of the 477 VRD training images referring to object class 'glasses' 
# began by filtering these 477 images to find the subset that had 
# relationships involving 'glasses' that were NOT a member of the set of
# filtering conditions. The objective was to find a (hopefully) small
# subset of images where (hopefully) references to 'glasses' referred only
# to *drinking glasses* as opposed to *eyeglasses*. The result set of images
# satisfying the filtering conditions contained 196 images. These were
# each viewed individually, along with their annotations, to idenfify and
# confirm the precise meaning of the object class label 'glasses' in each
# individual visual relationship annotation.

# Having analysed the result set of 196 images satisfying the filtering
# conditions, we identified 4 categories of exceptions where the 
# annotations of the image either did NOT refer exclusively to drinking 
# glasses, or DID refer exclusively to drinking glasses but had additional
# special problems that require special, annotation-specific fixes.
# These 4 exception categories (cases) are defined next.

# exception case 1: 
# This subset of images satisfied the filtering conditions and yet,
# nonetheless, were deemed to use the object class 'glasses' to refer to
# *eyeglasses* exclusively. Hence, the annotations referring to 'glasses'
# associated with these images can be considered *correct* and require
# no customisation.

# exception case 2:
# This subset of images satisfied the filtering conditions but were
# found to use object class 'glasses' to refer to BOTH *eyeglasses* and
# *drinking glass*. The specific annotations referring to *drinking glass*
# were identified and arrangements made to customise them to have them
# refer to a newly introduced object class of 'drinking glass'.

# exception case 3:
# This subset of images satisfied the filtering conditions but were
# found to use the object class 'glasses' in miscellaneous ways to refer to
# diverse types of object which, while made of glass, were neither
# *eyeglasses* nor *drinking glasses*.  The specific annotations in question
# were identified and arrangements made to either remove them from the
# annotations for the corresponding image or switch the object class to
# the correct object class.

# exception case 4:
# This subset of images satisfied the filtering conditions and were found
# to use object class 'glasses' to refer to *drinking glasses* exclusively,
# but there also found to have additional special issues that required
# individual special fixes (typically the rebuilding of a bad bbox for the
# 'glasses' (aka drinking glasses)).

# remaining subset of images:
# If one takes the full result set of 196 images that satisfied the
# filtering conditions and removes the subsets of images belonging to
# exception cases (1), (2), (3) and (4), one (in theory) is left with a
# remainging subset of images that use object class 'glasses' exclusively
# to refer to *drinking glasses* and which have no additional special
# issues to be fixed in relation to the glasses. This remaining subset of 
# images can thus be processed enmass, using a *global switch* strategy,
# to change object class 'glasses' to the newly introduced object class
# 'drinking glass'. We don't need to refer to individual annotations, we
# can simply write a function to look at the annotations for these images
# and change all instances of object class 'glasses' to object class
# 'drinking glass'.

# The lists of image names defined below for exception cases (1), (2), (3)
# and (4) are designed to facilitate identification of the *remainging subset*
# of images, by allowing the result set of 196 image names to be filtered
# automatically in order to yield the image names belonging to this
# *remaining subset* destined for correction via an enmass correction
# strategy rather than by specifying the specific annotations that require
# fixing.


glasses_exceptions_case_1 = [

    
'5570342667_f308a479f8_b.jpg',
'409555309_2714afe06f_b.jpg',
'4388565026_7556fdab14_o.jpg',
'54217796_6fcb4addaa_o.jpg',
'272277668_ef529278da_b.jpg',
'6947209003_db985f18af_b.jpg',
'7143450153_69cc8458ff_b.jpg',
'308042862_37894a5225_b.jpg',
'8116722351_7b6dc4d5a6_b.jpg',
'5402369803_8c559054d6_b.jpg',
'56030262_f38b94c3b6_b.jpg',
'4594556842_8cc62c5a17_b.jpg',
'9210367756_47509f417b_b.jpg',
'369251660_7345487e73_o.jpg',
'6024967819_730cd89e01_b.jpg',
'7213566078_b06cd36dfd_b.jpg',
'2308524757_c092cb07d8_b.jpg',
'8365091151_63acedf87e_b.jpg',
'9069081843_179e22e467_b.jpg',
'9165998511_3256601174_b.jpg',
'433955091_68f0116c65_b.jpg',
'8521013963_7e728c7588_b.jpg',
'6320506815_d815ee8b70_b.jpg',
'4129976857_93a5dc99de_o.jpg',
'9327085767_5ea42c4e0e_b.jpg',
'210903242_42780ed8fd_o.jpg',
'3623195683_56df9b50d3_b.jpg',
'3505702201_e1c0e8c586_o.jpg',
'5111487593_3a0150e853_b.jpg',
'2346032_d20f744136_b.jpg',
'6179115649_b110ff816c_b.jpg',
'8386698122_28a5318c03_b.jpg',
'9426740552_fe479a063e_b.jpg',
'3016201881_63f844f03a_o.jpg',
'2791980504_92fce6c14d_o.jpg',
'3555542512_940c38e8e0_b.jpg',
'5105786658_a9e051affc_b.jpg',
'341359951_23d185b673_o.jpg',
'3826260144_93654826ae_b.jpg',
'7713525964_0583997dd8_b.jpg',
'5002139641_1c41e3fa2c_b.jpg',
'6610791839_4f408b20af_b.jpg',
'4701704095_8cb324128b_b.jpg',
'3654652093_12b94463da_b.jpg',
'5415133934_408cd920e5_b.jpg',
'2714495032_0c9c21fbb3_b.jpg',
'1306025734_7cab73584b_b.jpg',
'466479036_df4e7b9352_b.jpg',
'6085367573_1821cfb58b_b.jpg',
'1474858933_512c81f463_b.jpg'
    
]

glasses_exceptions_case_2 = [

'5813297357_f210a455f9_b.jpg',
'508537639_845774e06a_b.jpg',
'3480657171_7719a4d68b_o.jpg',
'5422110190_756079d6bb_b.jpg',   
'255923222_d34e9a95f8_b.jpg',
'9514068596_48a19398c4_b.jpg',    
'8865584540_b5307795be_b.jpg',
'4759889504_460d484166_b.jpg',   
'9064793840_2a46a84f32_b.jpg',
'8719406627_d9142f9819_b.jpg',
'9176487150_b8bf04f8f3_b.jpg',
'9719081589_80dda07912_b.jpg',
'9449145978_f5715dcdb3_b.jpg',
'5819097734_c902914019_b.jpg',
'8171285264_cb15753997_b.jpg',
'4773542234_96254a4925_b.jpg',
'2351728863_2e3e576ea9_b.jpg',
'8349546940_55540c4f7c_b.jpg',
'10256449426_a8fbfc2bbc_b.jpg',
'8174126404_192257d449_b.jpg',
'6547507981_d3afb6b487_b.jpg',
'8588631266_ca299dd629_b.jpg',
'337505168_903fa2bb0d_b.jpg',
'9046906485_f3b1798e23_b.jpg',
'8430554096_17aef4753f_b.jpg',
'10187379915_e710e52c6d_b.jpg',
'10167338654_1a10aa72a0_b.jpg',
'8130686043_5b5424f47e_b.jpg'

]

glasses_exceptions_case_3 = [

'325477317_dafb06abc1_o.jpg',   
'5465451880_49cf298b5f_b.jpg',
'4684544925_3845f0b23a_b.jpg',   
'4388038953_c5cd22734f_o.jpg',
'3762322850_5f263f7efe_b.jpg',
'3743759994_4d1a980068_b.jpg',
'3376388630_bbda7f439c_b.jpg',
'9357336385_eed5fe232a_b.jpg',
'9420171433_0c28ed5369_b.jpg',
'8511176011_216b8626b0_b.jpg',
'94889859_38289e9d5c_o.jpg',
'4635800439_0701fb0132_b.jpg',
'451204248_e8ef6f0f15_b.jpg',
'4793736598_5d0ee8e418_b.jpg',
'6636124103_e0950d9ef8_b.jpg',
'3317392680_c573b9c0f8_b.jpg',
'2359293784_9f4ac7c9ab_b.jpg',
'4625274192_c9f66750f0_o.jpg'

]

glasses_exceptions_case_4 = [

'6673134881_ae78aa7c18_b.jpg',
'8447275481_a5470551fe_b.jpg',
'172636826_3487ce015e_o.jpg',
'139985421_8f1c4372fa_b.jpg',
'3802818240_6cacb58749_o.jpg',
'77398730_1811090b82_b.jpg',
'8315089900_910dfa2410_b.jpg',
'9692771432_7e9ba08cec_b.jpg',
'2413768065_b80fd4b342_b.jpg',
'5075675963_d2c1e7973f_b.jpg',
'6551396507_db2104f849_b.jpg',
'10180021573_049323338d_b.jpg',
'4151866555_0fe17b8737_o.jpg',
'4317088519_c7a9b42884_b.jpg',
'8859001381_57a720a602_b.jpg',
'9633163875_15c2967389_b.jpg'

]




