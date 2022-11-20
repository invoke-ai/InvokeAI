import BoundingBoxSettings from './BoundingBoxSettings/BoundingBoxSettings';
import InpaintReplace from './InpaintReplace';

export default function InpaintingSettings() {
  return (
    <>
      <InpaintReplace />
      <BoundingBoxSettings />
    </>
  );
}
