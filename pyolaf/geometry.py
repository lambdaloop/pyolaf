#!/usr/bin/env ipython

import numpy as np

def LFM_computeTube2MLA(lensPitch, mla2sensor, deltaOT, objRad, ftl):

    # tube2mla satisfies 3 conditions:
    # 1. matching effective f number with the
    # 2. effective tubeLensRadius
    # 3. the above when focused on the MLA

    # solve quadratic equation
    a = ftl*lensPitch/(2*mla2sensor)
    # b = delta_ot*objRad - ftl*objRad - mla2sensor*ftl;
    b = deltaOT*objRad - ftl*objRad
    c = -ftl*deltaOT*objRad

    delta_equation = (b**2 - 4*a*c)**(1/2)

    tube2mla = [(-b+delta_equation)/(2*a), (-b-delta_equation)/(2*a)]
    tube2mla = [i for i in tube2mla if i > 0]

    return tube2mla

import yaml

def LFM_setCameraParams(configFile, newSpacingPx):

    # set LFM parameters:
    #### objective params
    # M-> objective magnification
    # NA-> objective aperture
    # ftl-> focal length of tube lens (only for microscopes)

    #### sensor
    # lensPitch-> lenslet pitch
    # pixelPitch-> sensor pixel pitch

    #### MLA params
    # gridType-> microlens grid type: "reg" -> regular grid array; "hex" -> hexagonal grid array
    # focus-> microlens focus: "single" -> all ulens in the array have the same focal length; 'multi' -> 3 mixed focal lenghts in a hex grid
    # fml-> focal length of the lenslets
    # uLensMask-> 1 when there is no space between ulenses (rect shape); 0 when there is space between ulenses (circ aperture)

    #### light characteristics
    # n-> refraction index (1 for air)
    # wavelenght-> wavelenght of the the emission light

    ##### distances
    # plenoptic-> plenoptic type: "1" for the original LFM configuration (tube2mla = ftl); "2" for defocused LFM (tube2mla != ftl) design
    # tube2mla-> distance between tube lens and MLA (= ftl in original LFM design)
    # mla2sensor-> distance between MLA and sensor

    with open(configFile, "r") as f:
        Camera = yaml.safe_load(f)

    # if isinstance(Camera["fm"], list):
    #     Camera["fm"] = [float(val) for val in Camera["fm"]]
    #     Camera["fm"] = float(sum(Camera["fm"]))

    # compute extra LFM configuration specific parameters
    # for regular grid we only need to compute 1/4 of the psf pattens, due to symmetry
    if Camera["gridType"] == "reg":
        Camera["range"] = "quarter"
    elif Camera["gridType"] == "hex":
        Camera["range"] = "full"

    Camera["fobj"] = Camera["ftl"]/Camera["M"]  ## focal length of objective lens
    Camera["Delta_ot"] = Camera["ftl"] + Camera["fobj"] ## obj2tube distance

    # ulens spacing = ulens pitch
    spacingPx = Camera["lensPitch"]/Camera["pixelPitch"]
    if newSpacingPx == "default":
        newSpacingPx = spacingPx

    Camera["spacingPx"] = spacingPx
    Camera["newSpacingPx"] = newSpacingPx  # artificial spacing, usually lower than spacingPx for speed up
    Camera["newPixelPitch"] = Camera["lensPitch"]/newSpacingPx

    Camera["k"] = 2 * np.pi * Camera["n"] / Camera["WaveLength"] ## wave number

    objRad = Camera["fobj"] * Camera["NA"] # objective radius

    if Camera["plenoptic"] == 2:

        obj2tube = Camera["fobj"] + Camera["ftl"]  ## objective to tl distance

        # check if tube2mla is provided by user, otherwise compute it s.t. the F number matching condition is satisfied
        if Camera["tube2mla"] == 0:
            if Camera["mla2sensor"] == 0:
                raise ValueError("At least one of the 'tube2mla' or 'mla2sensor' distances need to be provided in the - plenoptic == 2 - case")
            else:
                tube2mla = LFM_computeTube2MLA(Camera["lensPitch"], Camera["mla2sensor"], obj2tube, objRad, Camera["ftl"])
                Camera["tube2mla"] = tube2mla

        dot = Camera["ftl"] * Camera["tube2mla"] / (Camera["tube2mla"] - Camera["ftl"])  ## depth focused on the mla by the tube lens (object side of tl)
        dio = obj2tube - dot  ## image side of the objective
        dof = Camera["fobj"] * dio / (dio - Camera["fobj"])  ## object side of the objective -> dof is focused on the MLA
        if np.isnan(dof):
            dof = Camera["fobj"]

        # M_mla = tube2mla/dof
        M_mla = Camera["M"]  ## in 4f systems magnification does not change with depth ?
        tubeRad = (dio - obj2tube) * objRad / dio

        # if tube2mla is known and mla2sensor has to be retrived s.t. F number matching condition is satisfied
        if Camera["mla2sensor"] == 0:
            mla2sensor = tube2mla * Camera["lensPitch"] / 2 / tubeRad
            Camera["mla2sensor"] = mla2sensor

    elif Camera["plenoptic"] == 1:

        if Camera["tube2mla"] == 0:
            Camera["tube2mla"] = Camera["ftl"]

        if Camera["mla2sensor"] == 0:
            Camera["mla2sensor"] = Camera["fm"]

        tubeRad = objRad
        dof = Camera["fobj"]  ## object side of the objective -> dof is focused on the mla
        M_mla = Camera["M"]  # magnification up to the mla position

    # Write extra computed parameters to Camera dictionary
    uRad =  tubeRad * Camera["mla2sensor"] / Camera["tube2mla"]
    offsetFobj = dof - Camera["fobj"]

    Camera["objRad"] = objRad
    Camera["uRad"] = uRad
    Camera["tubeRad"] = tubeRad
    Camera["dof"] = dof
    Camera["offsetFobj"] = offsetFobj
    Camera["M_mla"] = M_mla

    return Camera



def LFM_processWhiteImage(WhiteImage, spacingPx, gridType, DebugBuildGridModel):

    GridModelOptions = {}
    GridModelOptions['ApproxLensletSpacing'] = spacingPx   # lensPitch / pixelPitch;
    GridModelOptions['Orientation'] = 'horz'
    GridModelOptions['FilterDiskRadiusMult'] = 1/3
    GridModelOptions['CropAmt'] = 30 # changed from 30, gives better results
    GridModelOptions['SkipStep'] = 10
    GridModelOptions['Precision'] = 'single'

    # Find grid params
    LensletGridModel, gridCoords = LFM_BuildLensletGridModel(
        WhiteImage, gridType, GridModelOptions, DebugBuildGridModel )

    return LensletGridModel, gridCoords

from scipy import ndimage
from scipy.spatial import Delaunay, cKDTree
from skimage.morphology import disk

from scipy.signal import convolve2d
from scipy.spatial import Delaunay, distance
from skimage import filters
from skimage.feature import peak_local_max
from skimage.morphology import disk
from scipy import signal


def LFM_BuildLensletGridModel(WhiteImg, gridType, GridModelOptions, DebugDisplay=False):

    #---Defaults---
    # GridModelOptions = LFDefaultField( 'GridModelOptions', 'Precision', 'single' );
    # DebugDisplay = LFDefaultVal( 'DebugDisplay', false );

    #---Optionally rotate for vertically-oriented grids---
    if( GridModelOptions['Orientation'].lower() == 'vert' ):
        # this is flipped to be consistent with matlab code
        # (images are loaded transposed in matlab for some reason)
        WhiteImg = WhiteImg.T

    # Try locating lenslets by convolving with a disk
    # Also tested Gaussian... the disk seems to yield a stronger result
    h = np.zeros(WhiteImg.shape, dtype=GridModelOptions['Precision'])
    hr = disk(int(GridModelOptions['ApproxLensletSpacing'] * GridModelOptions['FilterDiskRadiusMult']))
    hr = hr / hr.max()

    print('Filtering...')
    # Convolve using fft
    WhiteImgConv = signal.fftconvolve(WhiteImg, hr, mode='same')
    WhiteImgConv = (WhiteImgConv - WhiteImgConv.min()) / np.abs(WhiteImgConv.max() - WhiteImgConv.min())


    print('Finding Peaks...')
    # Find peaks in convolution... ideally these are the lenslet centers
    Peaks = peak_local_max(WhiteImgConv, exclude_border=GridModelOptions['CropAmt'])
    PeakIdxY, PeakIdxX = Peaks.T

    # # Crop to central peaks; eliminates edge effects
    # InsidePts = np.where((PeakIdxY >= GridModelOptions["CropAmt"]) &
    #                      (PeakIdxY < (np.shape(WhiteImg)[0] - GridModelOptions["CropAmt"])) &
    #                      (PeakIdxX >= GridModelOptions["CropAmt"]) &
    #                      (PeakIdxX < (np.shape(WhiteImg)[1] - GridModelOptions["CropAmt"])))
    # PeakIdxY = PeakIdxY[InsidePts]
    # PeakIdxX = PeakIdxX[InsidePts]

    #---Form a Delaunay triangulation to facilitate row/column traversal---
    # Triangulation = Delaunay(np.column_stack((PeakIdxY, PeakIdxX)))
    PeakRef = np.column_stack([PeakIdxX, PeakIdxY])
    tree = cKDTree(PeakRef)

    # ---Traverse rows and columns of lenslets, collecting stats---
    if DebugDisplay:
        plt.figure()
        plt.imshow(WhiteImg, cmap='gray')
        plt.title('Lenslets Centers')
        plt.show(block=False)
        # plt.hold(True)

    #--Traverse vertically--
    print('Vertical fit...')
    YStart = GridModelOptions["CropAmt"]*2 + 1
    YStop = np.shape(WhiteImg)[0] - GridModelOptions["CropAmt"]*2 - 1

    RecPtsY = []
    LineFitY = []
    for XStart in range(GridModelOptions["CropAmt"]*2, np.shape(WhiteImg)[1]-GridModelOptions["CropAmt"]*2, GridModelOptions["SkipStep"]):
        CurPos = [XStart, YStart]
        pts = []
        while True:
            dist, ClosestLabel = tree.query(CurPos)
            ClosestPt = PeakRef[ClosestLabel]
            # print(ClosestPt)
            pts.append(ClosestPt)
            if DebugDisplay:
                plt.plot(ClosestPt[0], ClosestPt[1], 'r.')
            CurPos = np.copy(ClosestPt)
            CurPos[1] = round(CurPos[1] + GridModelOptions["ApproxLensletSpacing"] * np.sqrt(3))
            if CurPos[1] > YStop:
                break
        pts = np.array(pts)
        #--Estimate angle for this most recent line--
        if len(pts) > 10:
            LineFitY.append(np.polyfit(pts[3:-3,1], pts[3:-3,0], 1))
            RecPtsY.append(pts[3:-3])

    if DebugDisplay:
        plt.draw()
        plt.pause(0.01)
        plt.show()

    #--Traverse horizontally--
    print('Horizontal fit...')
    XStart = GridModelOptions['CropAmt']*2 + 1
    XStop = WhiteImg.shape[1]-GridModelOptions['CropAmt']*2 - 1

    RecPtsX = []
    LineFitX = []
    for YStart in range(GridModelOptions['CropAmt']*2, WhiteImg.shape[0]-GridModelOptions['CropAmt']*2, GridModelOptions['SkipStep']):
        CurPos = np.array([XStart, YStart])
        pts = []
        while True:
            dist, ClosestLabel = tree.query(CurPos)
            ClosestPt = PeakRef[ClosestLabel]
            pts.append(ClosestPt)
            if DebugDisplay:
                plt.plot(ClosestPt[0], ClosestPt[1], 'y.')
            CurPos = np.copy(ClosestPt)
            CurPos[0] = round(CurPos[0] + GridModelOptions['ApproxLensletSpacing'])
            if CurPos[0] > XStop:
                break
        pts = np.array(pts)
        #--Estimate angle for this most recent line--
        if len(pts) > 10:
            LineFitX.append(np.polyfit(pts[3:-3,0], pts[3:-3,1], 1))
            RecPtsX.append(pts[3:-3])

    if DebugDisplay:
        plt.show(block=False)

    #--Trim ends to wipe out alignment, initial estimate artefacts--
    RecPtsY = RecPtsY[3:-3]
    RecPtsX = RecPtsX[3:-3]

    #--Estimate angle--
    SlopeX = np.mean(LineFitX, axis=0)[0]
    SlopeY = np.mean(LineFitY, axis=0)[0]

    AngleX = np.arctan2(-SlopeX, 1)
    AngleY = np.arctan2(SlopeY, 1)
    EstAngle = np.mean([AngleX, AngleY])

    #--Estimate spacing, assuming approx zero angle--
    # t = RecPtsY[:,:,1]
    # YSpacing = np.mean(np.diff(t, axis=1))/2 / (np.sqrt(3)/2)
    YSpacing = np.mean([np.mean(np.diff(row[:,1]))
                        for row in RecPtsY]) /2 / (np.sqrt(3)/2)

    XSpacing = np.mean([np.mean(np.diff(row[:,0])) for row in RecPtsX])

    #--Correct for angle--
    XSpacing = XSpacing / np.cos(EstAngle)
    YSpacing = YSpacing / np.cos(EstAngle)

    # --Build initial grid estimate, starting with CropAmt for the offsets--
    if gridType == 'reg':
        LensletGridModel = {
            'HSpacing': XSpacing,
            'VSpacing': XSpacing,
            'HOffset': GridModelOptions['CropAmt'],
            'VOffset': GridModelOptions['CropAmt'],
            'Rot': -EstAngle,
            'Orientation': GridModelOptions['Orientation'],
            'FirstPosShiftRow': 2
        }
        LensletGridModel['UMax'] = int(np.ceil((WhiteImg.shape[1] - GridModelOptions['CropAmt']*2) / XSpacing))
        LensletGridModel['VMax'] = int(np.ceil((WhiteImg.shape[0] - GridModelOptions['CropAmt']*2) / XSpacing))
    else:  # hexagonal grid
        LensletGridModel = {
            'HSpacing': XSpacing,
            'VSpacing': YSpacing*np.sqrt(3)/2,
            'HOffset': GridModelOptions['CropAmt'],
            'VOffset': GridModelOptions['CropAmt'],
            'Rot': -EstAngle,
            'Orientation': GridModelOptions['Orientation'],
            'FirstPosShiftRow': 2
        }
        LensletGridModel['UMax'] = int(np.ceil((WhiteImg.shape[1] - GridModelOptions['CropAmt']*2) / XSpacing))
        LensletGridModel['VMax'] = int(np.ceil((WhiteImg.shape[0] - GridModelOptions['CropAmt']*2) / (YSpacing*np.sqrt(3)/2)))

    GridCoords = LFBuildGrid(LensletGridModel, gridType)

    # --Find offset to nearest peak for each--
    GridCoordsX = GridCoords[..., 0].ravel()
    GridCoordsY = GridCoords[..., 1].ravel()
    BuildGridCoords = np.column_stack((GridCoordsX, GridCoordsY))

    d, ix = tree.query(BuildGridCoords)
    IdealPtCoords = PeakRef[ix]

    # --Estimate single offset for whole grid--
    EstOffset = np.median(IdealPtCoords - BuildGridCoords, axis=0)
    LensletGridModel['HOffset'] += EstOffset[0]
    LensletGridModel['VOffset'] += EstOffset[1]

    #---Remove crop offset / find top-left lenslet---
    NewVOffset = LensletGridModel["VOffset"] % LensletGridModel["VSpacing"]
    VSteps = round((LensletGridModel["VOffset"] - NewVOffset) / LensletGridModel["VSpacing"]) # should be a whole number

    VStepParity = VSteps % 2
    if (gridType == "hex"):
        if VStepParity == 1:
            LensletGridModel["HOffset"] += LensletGridModel["HSpacing"]/2
        NewHOffset = LensletGridModel["HOffset"] % (LensletGridModel["HSpacing"]/2)
        HSteps = round((LensletGridModel["HOffset"] - NewHOffset) / (LensletGridModel["HSpacing"]/2)) # should be a whole number
        HStepParity = HSteps % 2
        LensletGridModel["FirstPosShiftRow"] = 2 - HStepParity
    else:
        NewHOffset = LensletGridModel["HOffset"] % LensletGridModel["HSpacing"]
    #     HSteps = round((LensletGridModel["HOffset"] - NewHOffset) / LensletGridModel["HSpacing"]) # should be a whole number

    if DebugDisplay:
        plt.plot(LensletGridModel["HOffset"], LensletGridModel["VOffset"], 'ro')
        plt.plot(NewHOffset, NewVOffset, 'yx')
        plt.show()

    LensletGridModel["HOffset"] = NewHOffset
    LensletGridModel["VOffset"] = NewVOffset

    #---Finalize grid---
    LensletGridModel["UMax"] = np.floor((WhiteImg.shape[1]-LensletGridModel["HOffset"])/LensletGridModel["HSpacing"]) + 1
    LensletGridModel["VMax"] = np.floor((WhiteImg.shape[0]-LensletGridModel["VOffset"])/LensletGridModel["VSpacing"]) + 1

    GridCoords = LFBuildGrid(LensletGridModel, gridType)

    print("...Done.")

    return LensletGridModel, GridCoords


from scipy.spatial.transform import Rotation

def LFBuildGrid(LensletGridModel, gridType):
    RotCent = np.eye(3)
    RotCent[0:2,2] = [LensletGridModel['HOffset'], LensletGridModel['VOffset']]

    ToOffset = np.eye(3)
    ToOffset[0:2,2] = [LensletGridModel['HOffset'], LensletGridModel['VOffset']]

    r = Rotation.from_euler('Z', LensletGridModel['Rot'])
    R = ToOffset @ RotCent @ r.as_matrix() @ np.linalg.inv(RotCent)

    vv, uu = np.meshgrid(-1 + (np.arange(0, LensletGridModel['VMax'])) * LensletGridModel['VSpacing'],
                         -1 + (np.arange(0, LensletGridModel['UMax'])) * LensletGridModel['HSpacing'])

    if gridType == 'hex':
        uu[LensletGridModel['FirstPosShiftRow']::2,:] += 0.5 * LensletGridModel['HSpacing']

    GridCoords = np.column_stack((uu.ravel(order='C'), vv.ravel(order='C'), np.ones(vv.size)))
    GridCoords = np.dot(R, GridCoords.T).T[:,0:2]
    GridCoords = GridCoords.reshape([int(LensletGridModel['VMax']),
                                     int(LensletGridModel['UMax']), 2],
                                    order="F")

    return GridCoords

def LFM_setGridModel(SpacingPx, FirstPosShiftRow, UMax, VMax, HOffset, VOffset, Rot, Orientation, gridType):

    # todo: defaults
    if(gridType == 'hex'):
        Spacing = [SpacingPx*np.cos(np.deg2rad(30)), SpacingPx]
        Spacing = np.ceil(Spacing).astype(int)
        Spacing = np.ceil(Spacing/2)*2
    elif(gridType == 'reg'):
        Spacing = [SpacingPx, SpacingPx]

    LensletGridModel = {}
    LensletGridModel['HSpacing'] = Spacing[1]
    LensletGridModel['VSpacing'] = Spacing[0]
    LensletGridModel['HOffset'] = HOffset
    LensletGridModel['VOffset'] = VOffset
    LensletGridModel['Rot'] = Rot
    LensletGridModel['UMax'] = UMax
    LensletGridModel['VMax'] = VMax
    LensletGridModel['Orientation'] = Orientation
    LensletGridModel['FirstPosShiftRow'] = FirstPosShiftRow

    return LensletGridModel

def LFM_computeResolution(LensletGridModel, TextureGridModel, Camera, depthRange, depthStep):
    # Compute sensor resolution
    # Number of pixels behind a lenslet
    NspacingLenslet = np.array([LensletGridModel['VSpacing'], LensletGridModel['HSpacing']])
    NspacingTexture = np.array([TextureGridModel['VSpacing'], TextureGridModel['HSpacing']])

    # Corresponding sensor/tex resolution
    if(Camera['gridType'] == 'hex'):
        sensorRes = [Camera['lensPitch']*np.cos(np.deg2rad(30))/NspacingLenslet[0], Camera['lensPitch']/NspacingLenslet[1]]
        Nnum = [np.max(NspacingLenslet) + 1, np.max(NspacingLenslet) + 1]
        TexNnum = [np.max(NspacingTexture) + 1, np.max(NspacingTexture) + 1]
    if(Camera['gridType'] == 'reg'):
        sensorRes = [Camera['lensPitch']/NspacingLenslet[0], Camera['lensPitch']/NspacingLenslet[1]]
        Nnum = NspacingLenslet.copy()
        TexNnum = NspacingTexture.copy()

    Nnum = Nnum + (1-np.mod(Nnum,2))
    TexNnum = TexNnum + (1-np.mod(TexNnum,2))

    # Size of a voxel in micrometers. (superResFactor is a factor of lensletResolution)
    # When superResFactor == 1, we reconstruct at lenslet resolution
    # When superResFactor == Nnum, we reconstruct at sensor resolution
    # texScaleFactor = superResFactor./Nnum;

    # make sure the superResFactor produces an odd number of voxels per
    # repetition patch (in front of a mlens)
    # TexNnum = floor(texScaleFactor(1)*Nnum);
    # TexNnum = TexNnum + (1-mod(TexNnum,2));
    texScaleFactor = TexNnum/Nnum

    texRes = sensorRes/(texScaleFactor*Camera['M'])
    texRes = np.append(texRes, depthStep)

    # Compute mask for sensor/texture to avoid overlap during convolution (for hex grid)
    # mask for the patches behind a lenslet
    sensMask = LFM_computePatchMask(NspacingLenslet, Camera['gridType'],
                                    sensorRes, Camera['uRad'], Nnum)

    # texture mask (different of sensor mask when tex/image resolutions are decoupled)
    texMask = LFM_computePatchMask(NspacingTexture, Camera['gridType'], texRes,
                                   TexNnum[0]*texRes[0]/2, TexNnum)

    # Set up a struct containing the resolution related info
    Resolution = {'Nnum': Nnum,
                  'Nnum_half': np.ceil(Nnum/2).astype(int),
                  'TexNnum': TexNnum,
                  'TexNnum_half': np.ceil(TexNnum/2).astype(int),
                  'sensorRes': sensorRes,
                  'texRes': texRes,
                  'sensMask': sensMask,
                  'texMask': texMask,
                  'depthStep': depthStep,
                  'depthRange': depthRange,
                  'depths': np.arange(depthRange[0], depthRange[1] + depthStep, depthStep),
                  'texScaleFactor': texScaleFactor,
                  'maskFlag': Camera['uLensMask'],
                  'NspacingLenslet': NspacingLenslet,
                  'NspacingTexture': NspacingTexture}
    return Resolution

def LFM_computePatchMask(Nspacing, gridType, pixelSize, patchRad, Nnum):
    print(Nspacing, gridType, pixelSize, patchRad, Nnum)

    ysensorspace = np.arange(-np.floor(Nnum[0]/2), np.floor(Nnum[0]/2)+1)
    xsensorspace = np.arange(-np.floor(Nnum[1]/2), np.floor(Nnum[1]/2)+1)
    x, y = np.meshgrid(pixelSize[0]*ysensorspace, pixelSize[1]*xsensorspace)
    mask = 1*(np.sqrt(x*x+y*y) < patchRad)

    # Resolve for holes and overlaps
    mask = LFM_fixMask(mask, Nspacing, gridType)

    return mask

from scipy.signal import convolve2d


def LFM_fixMask(mask, NewLensletSpacing, gridType):
    # This function corrects a mask for holes and overlaps
    # Uncomment plot related lines for a visual hint

    trial_space = np.zeros((3 * mask.shape[0], 3 * mask.shape[1]))
    r_center = np.ceil(trial_space.shape[0] / 2).astype(int)
    c_center = np.ceil(trial_space.shape[1] / 2).astype(int)
    rs = NewLensletSpacing[0]
    cs = NewLensletSpacing[1]

    if (gridType == 'hex'):
        for a in [r_center - rs, r_center + rs]:
            for b in [c_center - round(cs/2), c_center + round(cs/2)]:
                trial_space[a-1, b-1] = 1
        for a in [r_center]:
            for b in [c_center - cs, c_center, c_center + cs]:
                trial_space[a-1, b-1] = 1
    elif (gridType == 'reg'):
        for a in [r_center - rs, r_center, r_center + rs]:
            for b in [c_center - cs, c_center, c_center + cs]:
                trial_space[a-1, b-1] = 1

    r, c = mask.shape[:2]
    space = convolve2d(trial_space, mask, mode='same')
    space_center = space[r:r*2, c:c*2]

    newMask = mask.copy()
    for i in range(r):
        for j in range(c):
            # fix holes
            if(space_center[i,j] == 0):
                newMask[i,j] = 1
                space = convolve2d(trial_space, newMask, mode='same')
                space_center = space[r:r*2, c:c*2]

            # fix overlap
            elif(space_center[i,j] == 2):
                newMask[i,j] = 0
                space = convolve2d(trial_space, newMask, mode='same')
                space_center = space[r:r*2, c:c*2]

    return newMask
def LFM_computeLensCenters(LensletGridModel, TextureGridModel, sensorRes, focus, gridType):

    centersPixels = LFBuildGrid(LensletGridModel, gridType)
    centerOfSensor = np.round(np.array(centersPixels.shape[:2])/2.0+0.01).astype(int) - 1

    if focus == 'multi':
        centersPixels = LFM_addLensTypes(centersPixels, centerOfSensor)

    # Note: new_centers_pixels contains x coords in dim 1 and y coords in
    # dim 2; so we interchange those to account for the convention:
    # 1 = vertical = row = y = aa

    # lenslets centers on the sensor in pixels
    lensletCenters = {}
    lensletCenters['px'] = np.copy(centersPixels)
    centerOffset = [centersPixels[centerOfSensor[0], centerOfSensor[1], 1],
                    centersPixels[centerOfSensor[0], centerOfSensor[1], 0]]
    lensletCenters['offset'] = np.array(centerOffset) + 1

    lensletCenters['px'][:,:,0] = (centersPixels[:,:,1] - centerOffset[0])
    lensletCenters['px'][:,:,1] = (centersPixels[:,:,0] - centerOffset[1])

    # lenslets centers on the sensor in (um) needed for C++
    lensletCenters['metric'] = np.copy(centersPixels)
    lensletCenters['metric'][:,:,0] = lensletCenters['px'][:,:,0] * sensorRes[0]
    lensletCenters['metric'][:,:,1] = lensletCenters['px'][:,:,1] * sensorRes[1]

    # centers of repetition texture patches in voxels (different of the lenslets center when sensor/texture resolutions are different)
    centersVoxels = LFBuildGrid(TextureGridModel, gridType)
    centerOfTexture = np.round(np.array(centersVoxels.shape[:2])/2+0.01).astype(int) - 1

    centerOffset = [centersVoxels[centerOfTexture[0], centerOfTexture[1], 1], centersVoxels[centerOfTexture[0], centerOfTexture[1], 0]]
    lensletCenters['vox'] = np.zeros_like(centersVoxels)
    lensletCenters['vox'][:,:,0] = centersVoxels[:,:,1] - centerOffset[0]
    lensletCenters['vox'][:,:,1] = centersVoxels[:,:,0] - centerOffset[1]

    return lensletCenters

## TODO: indexing not thoroughly checked, may be wrong
def LFM_addLensTypes(lensCentersPx, matrixCenter):
    no_rows = lensCentersPx.shape[0]
    no_cols = lensCentersPx.shape[1]

    centersWithTypes = np.zeros((no_rows, no_cols, 3))
    centersWithTypes[:,:,0:2] = lensCentersPx

    centersWithTypes[matrixCenter[0], matrixCenter[1], 2] = 1

    center_line_first = 0
    if centersWithTypes[matrixCenter[0],0,0] < centersWithTypes[matrixCenter[0] + 1,0,0]:
        center_line_first = 1

    rows_11 = np.hstack([np.flipud(np.arange(matrixCenter[0]-2 ,-1, -2)), np.arange(matrixCenter[0],no_rows,2)])
    rows_12 = np.hstack([np.flipud(np.arange(matrixCenter[0]-1 ,-1, -2)), np.arange(matrixCenter[0] + 1,no_rows,2)])

    cols_11 = np.hstack([np.flipud(np.arange(matrixCenter[1]-3 ,-1, -3)), np.arange(matrixCenter[1],no_cols,3)])
    centersWithTypes[rows_11[:, None], cols_11, 2] = 1

    if (center_line_first):
        cols_12 = np.hstack([np.flipud(np.arange(matrixCenter[1]-2 ,-1, -3)), np.arange(matrixCenter[1] + 1,no_cols,3)])
        centersWithTypes[rows_12[:, None], cols_12, 2] = 1
    else:
        cols_12 = np.hstack([np.flipud(np.arange(matrixCenter[1]-1 ,-1, -3)), np.arange(matrixCenter[1] + 2,no_cols,3)])
        centersWithTypes[rows_12[:, None], cols_12, 2] = 1

    cols_21 = np.hstack([np.flipud(np.arange(matrixCenter[1]-2 ,-1, -3)), np.arange(matrixCenter[1] + 1,no_cols,3)])
    centersWithTypes[rows_11[:, None], cols_21, 2] = 2

    if (center_line_first):
        cols_22 = np.hstack([np.flipud(np.arange(matrixCenter[1]-1 ,-1, -3)), np.arange(matrixCenter[1] + 2,no_cols,3)])
        centersWithTypes[rows_12[:, None], cols_22, 2] = 2
    else:
        cols_22 = np.hstack([np.flipud(np.arange(matrixCenter[1] ,-1, -3)), np.arange(matrixCenter[1],no_cols,3)])
        centersWithTypes[rows_12[:, None], cols_22, 2] = 2

    centersWithTypes[centersWithTypes == 0] = 3

    return centersWithTypes


def LFM_computeGeometryParameters(Camera, WhiteImage, depthRange, depthStep, superResFactor, DebugBuildGridModel=False, imgSize=None):
    # Process white image to find the real lenslet centers
    if WhiteImage.size == 0: # for simulation purposes; build LensetGridModel from specs and imgSize (when a white image does not exist)
        # Build lenslet grid model (MLA descriptor)
        LensletGridModel = {}
        MLASize = np.ceil(np.array(imgSize) / Camera['newSpacingPx']) # effective size of the microlens array (in lenslets)
        LensletGridModel['UMax'] = MLASize[1] # no of lenslets
        LensletGridModel['VMax'] = MLASize[0]

        if Camera['gridType'] == 'hex':
            LensletGridModel['VSpacing'] = np.round(np.sqrt(3) / 2 * Camera['spacingPx'])
        else:
            LensletGridModel['VSpacing'] = np.round(Camera['spacingPx'])
        LensletGridModel['HSpacing'] = np.round(Camera['spacingPx'])
        LensletGridModel['FirstPosShiftRow'] = 1
        LensletGridModel['Orientation'] = 'horz'
        LensletGridModel['HOffset'] = 0 # mod(np.ceil(imgSize[1] / 2), Camera['newSpacingPx'])
        LensletGridModel['VOffset'] = 0 # mod(np.ceil(imgSize[0] / 2), Camera['newSpacingPx'])
    else:
        LensletGridModel, GridCoords = LFM_processWhiteImage(WhiteImage, Camera['spacingPx'], Camera['gridType'], DebugBuildGridModel)

    # Transform to integer centers position
    # create the desired grid model when choosing a new spacing between lenslets (in pixels); Camera.newSpacingPx determines the desired sensor resolution
    HOffset = 0
    VOffset = 0
    Rot = 0
    NewLensletGridModel = LFM_setGridModel(
        Camera['newSpacingPx'], LensletGridModel['FirstPosShiftRow'], LensletGridModel['UMax'], LensletGridModel['VMax'],
        HOffset, VOffset, Rot, LensletGridModel['Orientation'], Camera['gridType'])
    InputSpacing = np.array([LensletGridModel['HSpacing'], LensletGridModel['VSpacing']])
    NewSpacing = np.array([NewLensletGridModel['HSpacing'], NewLensletGridModel['VSpacing']])
    XformScale = NewSpacing / InputSpacing  # Notice the resized image will not be square
    NewOffset = np.round([LensletGridModel['HOffset'], LensletGridModel['VOffset']] * XformScale)
    NewLensletGridModel['HOffset'] = NewOffset[0]
    NewLensletGridModel['VOffset'] = NewOffset[1]

    # Compute the texture side grid model
    if superResFactor == 'default':
        superResFactor = Camera['newSpacingPx']

    # When superResFactor == 1, we reconstruct at lenslet resolution
    # When superResFactor == Nnum, we reconstruct at sensor resolution
    TextureGridModel = LFM_setGridModel(superResFactor, LensletGridModel['FirstPosShiftRow'], LensletGridModel['UMax'], LensletGridModel['VMax'],
                                            HOffset, VOffset, Rot, LensletGridModel['Orientation'], Camera['gridType'])

    # Compute resolution according to the new grid
    Resolution = LFM_computeResolution(NewLensletGridModel, TextureGridModel, Camera, depthRange, depthStep)
    Resolution['superResFactor'] = superResFactor
    print(f'Super resolution factor of: {Resolution["TexNnum"]}')
    print(f'Pix size: [{Resolution["sensorRes"][0]}, {Resolution["sensorRes"][1]}]')
    print(f'Vox size: [{Resolution["texRes"][0]}, {Resolution["texRes"][1]}, {Resolution["texRes"][2]}]')

    # Compute lenslets centers on the sensor and corresponding repetition patches centers in texture space (in voxels)
    NewLensletGridModel['FirstPosShiftRow'] = LensletGridModel['FirstPosShiftRow']
    TextureGridModel['FirstPosShiftRow'] = NewLensletGridModel['FirstPosShiftRow']
    LensletCenters = LFM_computeLensCenters(NewLensletGridModel, TextureGridModel, Resolution['sensorRes'], Camera['focus'], Camera['gridType'])

    return LensletCenters, Resolution, LensletGridModel, NewLensletGridModel




def LFM_computeTube2MLA(lensPitch, mla2sensor, deltaOT, objRad, ftl):

    # tube2mla satisfies 3 conditions:
    # 1. matching effective f number with the
    # 2. effective tubeLensRadius
    # 3. the above when focused on the MLA

    # solve quadratic equation
    a = ftl*lensPitch/(2*mla2sensor)
    # b = delta_ot*objRad - ftl*objRad - mla2sensor*ftl;
    b = deltaOT*objRad - ftl*objRad
    c = -ftl*deltaOT*objRad

    delta_equation = (b**2 - 4*a*c)**(1/2)

    tube2mla = [(-b+delta_equation)/(2*a), (-b-delta_equation)/(2*a)]
    tube2mla = [i for i in tube2mla if i > 0]

    return tube2mla

def LFM_setCameraParams(configFile, newSpacingPx):

    # set LFM parameters:
    #### objective params
    # M-> objective magnification
    # NA-> objective aperture
    # ftl-> focal length of tube lens (only for microscopes)

    #### sensor
    # lensPitch-> lenslet pitch
    # pixelPitch-> sensor pixel pitch

    #### MLA params
    # gridType-> microlens grid type: "reg" -> regular grid array; "hex" -> hexagonal grid array
    # focus-> microlens focus: "single" -> all ulens in the array have the same focal length; 'multi' -> 3 mixed focal lenghts in a hex grid
    # fml-> focal length of the lenslets
    # uLensMask-> 1 when there is no space between ulenses (rect shape); 0 when there is space between ulenses (circ aperture)

    #### light characteristics
    # n-> refraction index (1 for air)
    # wavelenght-> wavelenght of the the emission light

    ##### distances
    # plenoptic-> plenoptic type: "1" for the original LFM configuration (tube2mla = ftl); "2" for defocused LFM (tube2mla != ftl) design
    # tube2mla-> distance between tube lens and MLA (= ftl in original LFM design)
    # mla2sensor-> distance between MLA and sensor

    with open(configFile, "r") as f:
        Camera = yaml.safe_load(f)

    # if isinstance(Camera["fm"], list):
    #     Camera["fm"] = [float(val) for val in Camera["fm"]]
    #     Camera["fm"] = float(sum(Camera["fm"]))

    # compute extra LFM configuration specific parameters
    # for regular grid we only need to compute 1/4 of the psf pattens, due to symmetry
    if Camera["gridType"] == "reg":
        Camera["range"] = "quarter"
    elif Camera["gridType"] == "hex":
        Camera["range"] = "full"

    Camera["fobj"] = Camera["ftl"]/Camera["M"]  ## focal length of objective lens
    Camera["Delta_ot"] = Camera["ftl"] + Camera["fobj"] ## obj2tube distance

    # ulens spacing = ulens pitch
    spacingPx = Camera["lensPitch"]/Camera["pixelPitch"]
    if newSpacingPx == "default":
        newSpacingPx = spacingPx

    Camera["spacingPx"] = spacingPx
    Camera["newSpacingPx"] = newSpacingPx  # artificial spacing, usually lower than spacingPx for speed up
    Camera["newPixelPitch"] = Camera["lensPitch"]/newSpacingPx

    Camera["k"] = 2 * np.pi * Camera["n"] / Camera["WaveLength"] ## wave number

    objRad = Camera["fobj"] * Camera["NA"] # objective radius

    if Camera["plenoptic"] == 2:

        obj2tube = Camera["fobj"] + Camera["ftl"]  ## objective to tl distance

        # check if tube2mla is provided by user, otherwise compute it s.t. the F number matching condition is satisfied
        if Camera["tube2mla"] == 0:
            if Camera["mla2sensor"] == 0:
                raise ValueError("At least one of the 'tube2mla' or 'mla2sensor' distances need to be provided in the - plenoptic == 2 - case")
            else:
                tube2mla = LFM_computeTube2MLA(Camera["lensPitch"], Camera["mla2sensor"], obj2tube, objRad, Camera["ftl"])
                Camera["tube2mla"] = tube2mla

        dot = Camera["ftl"] * Camera["tube2mla"] / (Camera["tube2mla"] - Camera["ftl"])  ## depth focused on the mla by the tube lens (object side of tl)
        dio = obj2tube - dot  ## image side of the objective
        dof = Camera["fobj"] * dio / (dio - Camera["fobj"])  ## object side of the objective -> dof is focused on the MLA
        if np.isnan(dof):
            dof = Camera["fobj"]

        # M_mla = tube2mla/dof
        M_mla = Camera["M"]  ## in 4f systems magnification does not change with depth ?
        tubeRad = (dio - obj2tube) * objRad / dio

        # if tube2mla is known and mla2sensor has to be retrived s.t. F number matching condition is satisfied
        if Camera["mla2sensor"] == 0:
            mla2sensor = tube2mla * Camera["lensPitch"] / 2 / tubeRad
            Camera["mla2sensor"] = mla2sensor

    elif Camera["plenoptic"] == 1:

        if Camera["tube2mla"] == 0:
            Camera["tube2mla"] = Camera["ftl"]

        if Camera["mla2sensor"] == 0:
            Camera["mla2sensor"] = Camera["fm"]

        tubeRad = objRad
        dof = Camera["fobj"]  ## object side of the objective -> dof is focused on the mla
        M_mla = Camera["M"]  # magnification up to the mla position

    # Write extra computed parameters to Camera dictionary
    uRad =  tubeRad * Camera["mla2sensor"] / Camera["tube2mla"]
    offsetFobj = dof - Camera["fobj"]

    Camera["objRad"] = objRad
    Camera["uRad"] = uRad
    Camera["tubeRad"] = tubeRad
    Camera["dof"] = dof
    Camera["offsetFobj"] = offsetFobj
    Camera["M_mla"] = M_mla

    return Camera
