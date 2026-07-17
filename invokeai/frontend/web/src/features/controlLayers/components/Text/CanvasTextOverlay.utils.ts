type Size = {
  width: number;
  height: number;
};

type TextContentMetrics = {
  contentWidth: number;
  contentHeight: number;
};

type TextContainerPadding = {
  padding: number;
  extraLeftPadding: number;
  extraRightPadding: number;
};

type GetInitialCanvasTextEditorSizeArg = {
  contentMetrics: TextContentMetrics;
  textContainerData: TextContainerPadding;
  minSize: number;
};

type GetCanvasTextEditorEffectiveSizeArg = GetInitialCanvasTextEditorSizeArg & {
  measuredSize: Size | null;
};

export const getInitialCanvasTextEditorSize = ({
  contentMetrics,
  textContainerData,
  minSize,
}: GetInitialCanvasTextEditorSizeArg): Size => {
  return {
    width: Math.max(
      contentMetrics.contentWidth +
        textContainerData.padding * 2 +
        textContainerData.extraLeftPadding +
        textContainerData.extraRightPadding,
      minSize
    ),
    height: Math.max(contentMetrics.contentHeight + textContainerData.padding * 2, minSize),
  };
};

export const getCanvasTextEditorEffectiveSize = ({
  measuredSize,
  contentMetrics,
  textContainerData,
  minSize,
}: GetCanvasTextEditorEffectiveSizeArg): Size => {
  return measuredSize ?? getInitialCanvasTextEditorSize({ contentMetrics, textContainerData, minSize });
};
