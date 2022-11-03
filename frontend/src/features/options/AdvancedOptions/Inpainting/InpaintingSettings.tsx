import BoundingBoxSettings from './BoundingBoxSettings/BoundingBoxSettings';
import InpaintReplace from './InpaintReplace';
import ClearBrushHistory from './ClearBrushHistory';

export default function InpaintingSettings() {
  return (
    <>
      <InpaintReplace />
      <BoundingBoxSettings />
      <ClearBrushHistory />
    </>
  );
}
