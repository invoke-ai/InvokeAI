# This file is vendored from https://github.com/ShieldMnt/invisible-watermark
#
# `invisible-watermark` is MIT licensed as of August 23, 2025, when the code was copied into this repo.
#
# Why we vendored it in:
# `invisible-watermark` has a dependency on `opencv-python`, which conflicts with Invoke's dependency on
# `opencv-contrib-python`. It's easier to copy the code over than complicate the installation process by
# requiring an extra post-install step of removing `opencv-python` and installing `opencv-contrib-python`.

import struct
import uuid
import base64
import cv2
import numpy as np
import pywt


class WatermarkEncoder(object):
    def __init__(self, content=b""):
        seq = np.array([n for n in content], dtype=np.uint8)
        self._watermarks = list(np.unpackbits(seq))
        self._wmLen = len(self._watermarks)
        self._wmType = "bytes"

    def set_by_ipv4(self, addr):
        bits = []
        ips = addr.split(".")
        for ip in ips:
            bits += list(np.unpackbits(np.array([ip % 255], dtype=np.uint8)))
        self._watermarks = bits
        self._wmLen = len(self._watermarks)
        self._wmType = "ipv4"
        assert self._wmLen == 32

    def set_by_uuid(self, uid):
        u = uuid.UUID(uid)
        self._wmType = "uuid"
        seq = np.array([n for n in u.bytes], dtype=np.uint8)
        self._watermarks = list(np.unpackbits(seq))
        self._wmLen = len(self._watermarks)

    def set_by_bytes(self, content):
        self._wmType = "bytes"
        seq = np.array([n for n in content], dtype=np.uint8)
        self._watermarks = list(np.unpackbits(seq))
        self._wmLen = len(self._watermarks)

    def set_by_b16(self, b16):
        content = base64.b16decode(b16)
        self.set_by_bytes(content)
        self._wmType = "b16"

    def set_by_bits(self, bits=[]):
        self._watermarks = [int(bit) % 2 for bit in bits]
        self._wmLen = len(self._watermarks)
        self._wmType = "bits"

    def set_watermark(self, wmType="bytes", content=""):
        if wmType == "ipv4":
            self.set_by_ipv4(content)
        elif wmType == "uuid":
            self.set_by_uuid(content)
        elif wmType == "bits":
            self.set_by_bits(content)
        elif wmType == "bytes":
            self.set_by_bytes(content)
        elif wmType == "b16":
            self.set_by_b16(content)
        else:
            raise NameError("%s is not supported" % wmType)

    def get_length(self):
        return self._wmLen

    # @classmethod
    # def loadModel(cls):
    #     RivaWatermark.loadModel()

    def encode(self, cv2Image, method="dwtDct", **configs):
        (r, c, channels) = cv2Image.shape
        if r * c < 256 * 256:
            raise RuntimeError("image too small, should be larger than 256x256")

        if method == "dwtDct":
            embed = EmbedMaxDct(self._watermarks, wmLen=self._wmLen, **configs)
            return embed.encode(cv2Image)
        # elif method == 'dwtDctSvd':
        #     embed = EmbedDwtDctSvd(self._watermarks, wmLen=self._wmLen, **configs)
        #     return embed.encode(cv2Image)
        # elif method == 'rivaGan':
        #     embed = RivaWatermark(self._watermarks, self._wmLen)
        #     return embed.encode(cv2Image)
        else:
            raise NameError("%s is not supported" % method)


class WatermarkDecoder(object):
    def __init__(self, wm_type="bytes", length=0):
        self._wmType = wm_type
        if wm_type == "ipv4":
            self._wmLen = 32
        elif wm_type == "uuid":
            self._wmLen = 128
        elif wm_type == "bytes":
            self._wmLen = length
        elif wm_type == "bits":
            self._wmLen = length
        elif wm_type == "b16":
            self._wmLen = length
        else:
            raise NameError("%s is unsupported" % wm_type)

    def reconstruct_ipv4(self, bits):
        ips = [str(ip) for ip in list(np.packbits(bits))]
        return ".".join(ips)

    def reconstruct_uuid(self, bits):
        nums = np.packbits(bits)
        bstr = b""
        for i in range(16):
            bstr += struct.pack(">B", nums[i])

        return str(uuid.UUID(bytes=bstr))

    def reconstruct_bits(self, bits):
        # return ''.join([str(b) for b in bits])
        return bits

    def reconstruct_b16(self, bits):
        bstr = self.reconstruct_bytes(bits)
        return base64.b16encode(bstr)

    def reconstruct_bytes(self, bits):
        nums = np.packbits(bits)
        bstr = b""
        for i in range(self._wmLen // 8):
            bstr += struct.pack(">B", nums[i])
        return bstr

    def reconstruct(self, bits):
        if len(bits) != self._wmLen:
            raise RuntimeError("bits are not matched with watermark length")

        if self._wmType == "ipv4":
            return self.reconstruct_ipv4(bits)
        elif self._wmType == "uuid":
            return self.reconstruct_uuid(bits)
        elif self._wmType == "bits":
            return self.reconstruct_bits(bits)
        elif self._wmType == "b16":
            return self.reconstruct_b16(bits)
        else:
            return self.reconstruct_bytes(bits)

    def decode(self, cv2Image, method="dwtDct", **configs):
        (r, c, channels) = cv2Image.shape
        if r * c < 256 * 256:
            raise RuntimeError("image too small, should be larger than 256x256")

        bits = []
        if method == "dwtDct":
            embed = EmbedMaxDct(watermarks=[], wmLen=self._wmLen, **configs)
            bits = embed.decode(cv2Image)
        # elif method == 'dwtDctSvd':
        #     embed = EmbedDwtDctSvd(watermarks=[], wmLen=self._wmLen, **configs)
        #     bits = embed.decode(cv2Image)
        # elif method == 'rivaGan':
        #     embed = RivaWatermark(watermarks=[], wmLen=self._wmLen, **configs)
        #     bits = embed.decode(cv2Image)
        else:
            raise NameError("%s is not supported" % method)
        return self.reconstruct(bits)

    # @classmethod
    # def loadModel(cls):
    #     RivaWatermark.loadModel()


class EmbedMaxDct(object):
    def __init__(self, watermarks=[], wmLen=8, scales=[0, 36, 36], block=4):
        self._watermarks = watermarks
        self._wmLen = wmLen
        self._scales = scales
        self._block = block

    def encode(self, bgr):
        (row, col, channels) = bgr.shape

        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        for channel in range(2):
            if self._scales[channel] <= 0:
                continue

            ca1, (h1, v1, d1) = pywt.dwt2(yuv[: row // 4 * 4, : col // 4 * 4, channel], "haar")
            self.encode_frame(ca1, self._scales[channel])

            yuv[: row // 4 * 4, : col // 4 * 4, channel] = pywt.idwt2((ca1, (v1, h1, d1)), "haar")

        bgr_encoded = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return bgr_encoded

    def decode(self, bgr):
        (row, col, channels) = bgr.shape

        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        scores = [[] for i in range(self._wmLen)]
        for channel in range(2):
            if self._scales[channel] <= 0:
                continue

            ca1, (h1, v1, d1) = pywt.dwt2(yuv[: row // 4 * 4, : col // 4 * 4, channel], "haar")

            scores = self.decode_frame(ca1, self._scales[channel], scores)

        avgScores = list(map(lambda l: np.array(l).mean(), scores))

        bits = np.array(avgScores) * 255 > 127
        return bits

    def decode_frame(self, frame, scale, scores):
        (row, col) = frame.shape
        num = 0

        for i in range(row // self._block):
            for j in range(col // self._block):
                block = frame[
                    i * self._block : i * self._block + self._block, j * self._block : j * self._block + self._block
                ]

                score = self.infer_dct_matrix(block, scale)
                # score = self.infer_dct_svd(block, scale)
                wmBit = num % self._wmLen
                scores[wmBit].append(score)
                num = num + 1

        return scores

    def diffuse_dct_svd(self, block, wmBit, scale):
        u, s, v = np.linalg.svd(cv2.dct(block))

        s[0] = (s[0] // scale + 0.25 + 0.5 * wmBit) * scale
        return cv2.idct(np.dot(u, np.dot(np.diag(s), v)))

    def infer_dct_svd(self, block, scale):
        u, s, v = np.linalg.svd(cv2.dct(block))

        score = 0
        score = int((s[0] % scale) > scale * 0.5)
        return score
        if score >= 0.5:
            return 1.0
        else:
            return 0.0

    def diffuse_dct_matrix(self, block, wmBit, scale):
        pos = np.argmax(abs(block.flatten()[1:])) + 1
        i, j = pos // self._block, pos % self._block
        val = block[i][j]
        if val >= 0.0:
            block[i][j] = (val // scale + 0.25 + 0.5 * wmBit) * scale
        else:
            val = abs(val)
            block[i][j] = -1.0 * (val // scale + 0.25 + 0.5 * wmBit) * scale
        return block

    def infer_dct_matrix(self, block, scale):
        pos = np.argmax(abs(block.flatten()[1:])) + 1
        i, j = pos // self._block, pos % self._block

        val = block[i][j]
        if val < 0:
            val = abs(val)

        if (val % scale) > 0.5 * scale:
            return 1
        else:
            return 0

    def encode_frame(self, frame, scale):
        """
        frame is a matrix (M, N)

        we get K (watermark bits size) blocks (self._block x self._block)

        For i-th block, we encode watermark[i] bit into it
        """
        (row, col) = frame.shape
        num = 0
        for i in range(row // self._block):
            for j in range(col // self._block):
                block = frame[
                    i * self._block : i * self._block + self._block, j * self._block : j * self._block + self._block
                ]
                wmBit = self._watermarks[(num % self._wmLen)]

                diffusedBlock = self.diffuse_dct_matrix(block, wmBit, scale)
                # diffusedBlock = self.diffuse_dct_svd(block, wmBit, scale)
                frame[
                    i * self._block : i * self._block + self._block, j * self._block : j * self._block + self._block
                ] = diffusedBlock

                num = num + 1
