import { Button, Flex, Select } from "@invoke-ai/ui-library";
import { skipToken } from "@reduxjs/toolkit/query";
import { useAppSelector } from "app/store/storeHooks";
import { convertImageUrlToBlob } from "common/util/convertImageUrlToBlob";
import { useEditor } from "features/editImageModal/hooks/useEditor";
import { $imageName } from "features/editImageModal/store";
import { selectAutoAddBoardId } from "features/gallery/store/gallerySelectors";
import { useCallback,useEffect, useRef, useState } from "react";
import { useGetImageDTOQuery, useUploadImageMutation } from "services/api/endpoints/images";

export const EditorContainer = () => {
    const containerRef = useRef<HTMLDivElement>(null);
    const editor = useEditor({ containerRef });
    const [zoomLevel, setZoomLevel] = useState(100);
    const [cropInfo, setCropInfo] = useState<string>("");
    const [isInCropMode, setIsInCropMode] = useState(false);
    const [hasCrop, setHasCrop] = useState(false);
    const [aspectRatio, setAspectRatio] = useState<string>("free");
    const { data: imageDTO } = useGetImageDTOQuery($imageName.get() ?? skipToken);
    const autoAddBoardId = useAppSelector(selectAutoAddBoardId);

    const [uploadImage, { isLoading }] = useUploadImageMutation({ fixedCacheKey: 'editorContainer' });


    const loadImage = useCallback(async () => {
        if (!imageDTO) {
            console.error("Image not found");
            return;
        }
        const blob = await convertImageUrlToBlob(imageDTO.image_url);
        if (!blob) {
            console.error("Failed to convert image to blob");
            return;
        }
        await editor.loadImage(blob);
    }, [editor]);

    // Setup callbacks
    useEffect(() => {
        loadImage();
        editor.setCallbacks({
            onZoomChange: (zoom) => setZoomLevel(Math.round(zoom * 100)),
            onCropChange: (crop) => setCropInfo(`Crop: ${Math.round(crop.x)}, ${Math.round(crop.y)} - ${Math.round(crop.width)}x${Math.round(crop.height)}`),
            onImageLoad: () => {
                setCropInfo("");
                setIsInCropMode(false);
                setHasCrop(false);
            }
        });


    }, [editor, loadImage]);



    const handleStartCrop = () => {
        editor.startCrop();
        setIsInCropMode(true);
        // Apply current aspect ratio if not free
        if (aspectRatio !== "free") {
            const ratios: Record<string, number> = {
                "1:1": 1,
                "4:3": 4 / 3,
                "16:9": 16 / 9,
                "3:2": 3 / 2,
                "2:3": 2 / 3,
                "9:16": 9 / 16,
            };
            editor.setCropAspectRatio(ratios[aspectRatio]);
        }
    };

    const handleAspectRatioChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const newRatio = e.target.value;
        setAspectRatio(newRatio);

        if (newRatio === "free") {
            editor.setCropAspectRatio(undefined);
        } else {
            const ratios: Record<string, number> = {
                "1:1": 1,
                "4:3": 4 / 3,
                "16:9": 16 / 9,
                "3:2": 3 / 2,
                "2:3": 2 / 3,
                "9:16": 9 / 16,
            };
            editor.setCropAspectRatio(ratios[newRatio]);
        }
    };

    const handleApplyCrop = () => {
        editor.applyCrop();
        setIsInCropMode(false);
        setHasCrop(true);
        setCropInfo("");
        setAspectRatio("free");
    };

    const handleCancelCrop = () => {
        editor.cancelCrop();
        setIsInCropMode(false);
        setCropInfo("");
        setAspectRatio("free");
    };

    const handleResetCrop = () => {
        editor.resetCrop();
        setHasCrop(false);
    };

    const handleExport = async () => {
        try {
            const blob = await editor.exportImage("blob") as Blob;
            const file = new File([blob], "image.png", { type: "image/png" });

            await uploadImage({
                file,
                is_intermediate: false,
                image_category: 'user',
                board_id: autoAddBoardId === 'none' ? undefined : autoAddBoardId,
            }).unwrap();



        } catch (err) {
            console.error("Export failed:", err);
            if (err instanceof Error && err.message.includes("tainted")) {
                alert("Cannot export image: The image is from a different domain (CORS issue). To fix this:\n\n1. Load images from the same domain\n2. Use images from CORS-enabled sources\n3. Upload a local image file instead");
            } else {
                alert(`Export failed: ${  err instanceof Error ? err.message : String(err)}`);
            }
        }
    };

    return (
        <Flex w="full" h="full" flexDir="column">
            <Flex sx={{
                padding: "10px",
                background: "#f0f0f0",
                borderBottom: "1px solid #ccc",
                display: "flex",
                gap: "10px",
                alignItems: "center",
                flexWrap: "wrap"
            }}>


                <Flex sx={{ borderLeft: "1px solid #ccc", paddingLeft: "10px", display: "flex", gap: "10px", alignItems: "center" }}>
                    {!isInCropMode && (
                        <>
                            <Button onClick={handleStartCrop}>Start Crop</Button>
                            {hasCrop && <Button onClick={handleResetCrop}>Reset Crop</Button>}
                        </>
                    )}
                    {isInCropMode && (
                        <>
                            <Select
                                value={aspectRatio}
                                onChange={handleAspectRatioChange}
                                style={{ padding: "4px 8px" }}
                            >
                                <option value="free">Free</option>
                                <option value="1:1">1:1 (Square)</option>
                                <option value="4:3">4:3</option>
                                <option value="16:9">16:9</option>
                                <option value="3:2">3:2</option>
                                <option value="2:3">2:3 (Portrait)</option>
                                <option value="9:16">9:16 (Portrait)</option>
                            </Select>
                            <Button onClick={handleApplyCrop}>Apply Crop</Button>
                            <Button onClick={handleCancelCrop}>Cancel Crop</Button>
                        </>
                    )}
                </Flex>

                <Flex sx={{ borderLeft: "1px solid #ccc", paddingLeft: "10px" }}>
                    <Button onClick={() => editor.fitToContainer()}>Fit</Button>
                    <Button onClick={() => editor.resetView()}>Reset View</Button>
                    <Button onClick={() => editor.setZoom(editor.getZoom() * 1.2)}>Zoom In</Button>
                    <Button onClick={() => editor.setZoom(editor.getZoom() / 1.2)}>Zoom Out</Button>
                </Flex>

                <Button onClick={handleExport}>Export</Button>

                <Flex sx={{ marginLeft: "auto", display: "flex", gap: "20px", fontSize: "14px" }}>
                    <span>Zoom: {zoomLevel}%</span>
                    {cropInfo && <span>{cropInfo}</span>}
                    {hasCrop && <span style={{ color: "green" }}>✓ Crop Applied</span>}
                </Flex>
            </Flex>

            <Flex ref={containerRef} sx={{ flex: 1, position: "relative" }}>
                <Flex sx={{
                    position: "absolute",
                    bottom: "10px",
                    left: "10px",
                    background: "rgba(0,0,0,0.7)",
                    color: "white",
                    padding: "5px 10px",
                    borderRadius: "3px",
                    fontSize: "12px"
                }}>
                    <Flex>Mouse wheel: Zoom</Flex>
                    <Flex>Space + Drag: Pan</Flex>
                    {isInCropMode && <Flex>Drag crop box or handles to adjust</Flex>}
                    {isInCropMode && aspectRatio !== "free" && <Flex>Aspect ratio: {aspectRatio}</Flex>}
                </Flex>
            </Flex>
        </Flex>
    );
}

