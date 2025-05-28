
from PIL import Image, ImageChops, ImageDraw
imgFLR,imgFTB = Image.FLIP_LEFT_RIGHT,Image.FLIP_TOP_BOTTOM
class Agent:
    def __init__(self):
        pass
    def Solve(self, problem):
        if problem.problemType == '2x2':
            return self.rpmSolve2x2(problem)
        elif problem.problemType == '3x3':
            return self.rpmSolve3x3(problem)
        else:
            return 9

    def rpmSolve2x2(self, problem):
        imgA, imgB, imgC = [Image.open(problem.figures[key].visualFilename).convert('L') for key in ['A', 'B', 'C']]
        optionLs = [Image.open(problem.figures[str(i)].visualFilename).convert('L') for i in range(1, 7)]
        return max(enumerate(self.evalScore2x2(imgA, imgB, imgC, optionLs)), key=lambda x: x[1])[0] + 1

    def rpmSolve3x3(self, problem):
        imgA, imgB, imgC, imgD, imgE, imgF, imgG, imgH = [Image.open(problem.figures[key].visualFilename).convert('L') for key in
                                  ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']]
        optionLs = [Image.open(problem.figures[str(i)].visualFilename).convert('L') for i in range(1, 9)]
        return max(enumerate(self.evalScore3x3(imgA, imgB, imgC, imgD, imgE, imgF, imgG, imgH, optionLs)), key=lambda x: x[1])[0] + 1

    def evalScore2x2(self, imgA, imgB, imgC, optionLs):
        reflectTrans = self.combTransScores(self.transposeScore, imgA, imgB, imgC, optionLs)
        rotateTrans = self.combTransScores(self.rotateScore, imgA, imgB, imgC, optionLs)
        compDifference = self.combTransScores(self.compDifference, imgA, imgB, imgC, optionLs)
        imgFloodFill = self.combTransScores(self.evalScoreFill, imgA, imgB, imgC, optionLs)
        scoreLs = []
        for reft, rot, cmpDiff, imgFill in zip(reflectTrans, rotateTrans, compDifference, imgFloodFill):
            scoreCalc = 2 * reft + 0.75 * cmpDiff + rot + 0.75 * imgFill
            scoreLs.append(scoreCalc)
        return scoreLs

    def evalScore3x3(self, imgA, imgB, imgC, imgD, imgE, imgF, imgG, imgH, optionLs):
        scoreMatch,pixUnion = self.scoreMatch(imgA, optionLs), self.upd(imgA, imgB, imgC, imgD, imgE, imgF, imgG, imgH, optionLs)
        reflectTrans3x3,rotateTrans3x3 = self.reftScore3x3(imgA, imgB, imgC, imgG, optionLs), self.rotateScore3x3(imgA, imgC, imgG, optionLs)
        checkSymmetry = self.imgSplitScore(imgA, imgB, imgC, imgG, imgH, optionLs)
        diff1,diff2,diff3 = self.compDiffDC(imgC, imgF, optionLs),self.compDiffDC(imgG, imgH, optionLs),self.compDiffE(imgA, imgB, imgC, imgG, imgH, optionLs)
        combDiffSet = [d1 + d2 + d3 for d1, d2, d3 in zip(diff1, diff2,diff3)]
        scoreLs = []
        i = 0
        while i < len(scoreMatch):
            smh,pxu,reft3x3,rot3x3,symm,diffs = scoreMatch[i],pixUnion[i],reflectTrans3x3[i],rotateTrans3x3[i],checkSymmetry[i],combDiffSet[i]
            scoreCalc = 2 * diffs + smh + pxu + reft3x3 + rot3x3 + symm
            scoreLs.append(scoreCalc)
            i += 1
        i = 0
        if self.checkSameImage(imgA, imgB, 0.01) and self.checkSameImage(imgB, imgC, 0.01):
            scoreLs = []
            while i < len(scoreMatch):
                smh, pxu, reft3x3, rot3x3,symm= scoreMatch[i], pixUnion[i], reflectTrans3x3[i], rotateTrans3x3[i], checkSymmetry[i]
                scoreCalc = smh + pxu + reft3x3 + rot3x3 + symm
                scoreLs.append(scoreCalc)
                i += 1
        return scoreLs

    def combTransScores(self, transformationEvalScore, imgA, imgB, imgC, optionLs):
        es1,es2=transformationEvalScore(imgA, imgB, imgC, optionLs),transformationEvalScore(imgA, imgC, imgB, optionLs)
        zipScores = []
        for sc1,sc2 in zip(es1,es2):
            zipScores.append(sc1 + sc2)
        return zipScores

    def pixRatio(self, image):
        pixels = list(image.getdata())
        blk = pixels.count(0)
        wht = len(pixels) - blk
        return wht / max(blk,1)

    def imageFloodFill(self, image):
        ImageDraw.floodfill(image,(image.size[0] // 2, image.size[1] // 2), 0)
        return image

    def transMethodsReflectRotate(self, transpose, rotate):
        if transpose:
            return [(imgFLR, 0.05), (imgFTB, 0.18)]
        elif rotate:
            return [(rotate, 0.1)]
        return []

    def applytransMethodsReflectRotate(self, image, transMethod, transpose):
        if transpose:
            return image.transpose(transMethod)
        return image.rotate(transMethod)

    def compareTransImageOptions(self, newFigC, optionLs, threshold, evalScore):
        for ind, option in enumerate(optionLs):
            if self.checkSameImage(newFigC, option, threshold):
                evalScore[ind] = max(evalScore[ind], 1)
        return evalScore

    def transposeScore(self, imgA, imgB, imgC, optionLs):
        return self.refRotScore(imgA, imgB, imgC, optionLs, transpose=True)

    def rotateScore(self, imgA, imgB, imgC, optionLs):
        return self.refRotScore(imgA, imgB, imgC, optionLs, rotate=270)

    def refRotScore(self, imgA, imgB, imgC, optionLs, transpose=False, rotate=None):
        evalScore,transMethods = [0] * len(optionLs), self.transMethodsReflectRotate(transpose, rotate)
        for trans, threshold in transMethods:
            newFigA,newFigC = self.applytransMethodsReflectRotate(imgA, trans, transpose),self.applytransMethodsReflectRotate(imgC, trans, transpose)
            if self.checkSameImage(newFigA, imgB, threshold):
                evalScore = self.compareTransImageOptions(newFigC, optionLs, threshold, evalScore)
        return evalScore

    def compDifference(self, imgA, imgB, imgC, optionLs):
        return [1 if self.checkSameImage(ImageChops.difference(imgA, imgB), ImageChops.difference(imgC, option), 0.037) else 0 for option in optionLs]

    def evalScoreFill(self, imgA, imgB, imgC, optionLs):
        imgFillA, imgFillC, evalScore = self.imageFloodFill(imgA.copy()),self.imageFloodFill(imgC.copy()),[0] * len(optionLs)
        if self.pixRatio(imgB) >= self.pixRatio(imgA) or not self.checkSameImage(imgFillA, imgB, 0.05):
            return evalScore
        return [5 if self.checkSameImage(imgFillC, option, 0.05) else 0 for option in optionLs]

    def checkSameImage(self, imgOne, imgTwo, threshold):
        diffImg = ImageChops.difference(imgOne, imgTwo)
        return self.pixRatio(diffImg) < threshold

    def upd(self, imgA, imgB, imgC, imgD, imgE, imgF, imgG, imgH, optionLs):
        R1, R2 = self.bpd(imgA) + self.bpd(imgB) + self.bpd(imgC), self.bpd(imgD) + self.bpd(imgE) + self.bpd(imgF)
        R3, uniPixScore,rDensDiff, evalScore = self.bpd(imgG) + self.bpd(imgH),[],abs(R1 - R2),[0] * len(optionLs)
        for option in optionLs:
            R3Option = R3 + self.bpd(option)
            uniPixScore.append(self.uniScore(R1, R2, R3Option, rDensDiff))
        return uniPixScore if uniPixScore else evalScore

    def compDiffDC(self, imgC, imgF, optionLs):
        imgCPixRatio,imgFPixRatio, cfDiff = self.pixRatio(imgC),self.pixRatio(imgF), ImageChops.difference(imgC, imgF)
        diffCF,diffFC,evalScore = imgCPixRatio - imgFPixRatio,imgFPixRatio - imgCPixRatio, [0] * len(optionLs)
        if diffCF > 0.24:
            return self.diffScoreCalc(imgF, optionLs, cfDiff, imgFPixRatio, threshold = 0.34)
        elif diffFC > 0.28:
            return self.diffScoreCalc(imgF, optionLs, cfDiff, imgFPixRatio, threshold = -0.38)
        return evalScore

    def diffScoreCalc(self, imgF, optionLs, scoreDiff, refRatio, threshold):
        scoreDifferences = []
        for option in optionLs:
            ratDiff = (self.pixRatio(option) - refRatio) if threshold < 0 else (refRatio - self.pixRatio(option))
            if (threshold < 0 and ratDiff > abs(threshold)) or (0 < threshold < ratDiff):
                evalScore = self.scoreThresholds(ImageChops.difference(imgF, option), scoreDiff)
                scoreDifferences.append(evalScore)
            else:
                scoreDifferences.append(0)
        return scoreDifferences

    def scoreThresholds(self, diff1, diff):
        scoreThresholdMap = {15: 0.037, 10: 0.06, 5: 0.08, 3: 0.1, 2: 0.2}
        for sc, thres in scoreThresholdMap.items():
            if self.checkSameImage(diff1, diff, thres):
                return sc
        return 1

    def bpd(self, img):
        return list(img.getdata()).count(0) / len(list(img.getdata()))

    def uniScore(self, R1, R2, R3Option, rDensDiff):
        if rDensDiff < 0.002:
            return self.smallUniScoreDiff(R1, R3Option)
        elif rDensDiff < 0.012:
            return self.medUniScoreDiff(R1, R2, R3Option)
        else:
            return 0

    def smallUniScoreDiff(self, R1, R3Option):
        return 10 if abs(R3Option - R1) < 0.002 else 0

    def medUniScoreDiff(self, R1, R2, R3Option):
        if abs(R3Option - R1) < 0.037 and abs(R3Option - R2) < 0.037:
            return 8
        elif abs(R3Option - R1) < 0.05 and abs(R3Option - R2) < 0.05:
            return 4
        return 0

    def reftScore3x3(self, imgA, imgB, imgC, imgG, optionLs):
        imgALR,imgGLR, evalScore = imgA.transpose(imgFLR), imgG.transpose(imgFLR), [0] * len(optionLs)
        imgAPR, imgCPR, imgGPR = map(self.pixRatio, [imgA, imgC, imgG])
        if self.checkSameImage(imgALR, imgB, 0.03) and (imgCPR - imgAPR) > 0.12:
            return [15 if (self.pixRatio(op) - imgGPR) > 0.12 else 0 for op in optionLs]
        elif self.checkSameImage(imgALR, imgC, 0.05):
            return [1 if self.checkSameImage(imgGLR, option, 0.05) else 0 for option in optionLs]
        return evalScore

    def rotateScore3x3(self, imgA, imgC, imgG, optionLs):
        imgALR,imgARot,imgGRot,evalScore=imgA.transpose(imgFLR),imgA.rotate(270), imgG.rotate(270),[0] * len(optionLs)
        if self.checkSameImage(imgALR, imgC, 0.027):
            return evalScore
        if self.checkSameImage(imgARot, imgC, 0.027):
            return [8 if self.checkSameImage(imgGRot, option, 0.027) else 0 for option in optionLs]
        return evalScore

    def imgSplitScore(self, imgA, imgB, imgC, imgG, imgH, optionLs):
        evalScore = [0] * len(optionLs)
        def imgSplit(img1, img2, dir):
            img1h,img1w,img2h, img2w =  img1.height,img1.width,img2.height,img2.width
            if dir == 'flipLr':
                cl1, cr1 = img1.crop((0, 0, img1w // 2, img1h)), img1.crop((img1w // 2, 0, img1w, img1h))
                cl2, cr2 = img2.crop((0, 0, img2w // 2, img2h)), img2.crop((img2w // 2, 0, img2w, img2h))
                return self.checkSameImage(cl1, cr2, 0.12) and self.checkSameImage(cr1, cl2, 0.12)
            if dir == 'flipTd':
                t1, d1 = img1.crop((0, 0, img1w, img1h // 2)), img1.crop((0, img1h // 2, img1w, img1h))
                t2, d2 = img2.crop((0, 0, img2w, img2h // 2)), img2.crop((0, img2h // 2, img2w, img2h))
                return "upper" if self.checkSameImage(t1, t2,0.03) else "lower" if self.checkSameImage(d1, d2,0.03) else False
            return False

        if imgSplit(imgA, imgC, 'flipLr'):
            return [5 if imgSplit(imgG, option, 'flipLr') else 0 for option in optionLs]
        tdState = imgSplit(imgA, imgC, 'flipTd'), imgSplit(imgB, imgC, 'flipTd')
        if tdState == ("upper", "lower"):
            return [5 if imgSplit(imgG, option, 'flipTd') == "upper" and imgSplit(imgH, option, 'flipTd') == "lower" else 0 for option in optionLs]
        if tdState == ("lower", "upper"):
            return [5 if imgSplit(imgG, option, 'flipTd') == "lower" and imgSplit(imgH, option, 'flipTd') == "upper" else 0 for option in optionLs]
        return evalScore

    def scoreMatch(self, figChoice, optionLs):
        return [1 if self.checkSameImage(figChoice, option, 1) else 0 for option in optionLs]

    def compDiffE(self, imgA, imgB, imgC, imgG, imgH, optionLs):
        scoreDiffE, evalScore = [],[0] * len(optionLs)
        if self.checkSameImage(ImageChops.invert(ImageChops.difference(imgA, imgB)), imgC, 0.04):
            for option in optionLs:
                scoreDiffE.append(15 if self.checkSameImage(ImageChops.invert(ImageChops.difference(imgG, imgH)), option, 0.06) else 0)
        return scoreDiffE or evalScore