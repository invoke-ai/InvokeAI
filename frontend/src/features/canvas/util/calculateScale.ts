const calculateScale = (
  containerWidth: number,
  containerHeight: number,
  contentWidth: number,
  contentHeight: number,
  padding = 0.95
): number => {
  const scaleX = (containerWidth * padding) / contentWidth;
  const scaleY = (containerHeight * padding) / contentHeight;
  const scaleFit = Math.min(1, Math.min(scaleX, scaleY));
  return scaleFit;
};

export default calculateScale;
