import InvokeWorkarea from 'features/ui/components/common/InvokeWorkarea';
import ImageToImageContent from './ImageToImageContent';
import ImageToImageParameters from './ImageToImageParameters';

export default function ImageToImageWorkarea() {
  return (
    <InvokeWorkarea parametersPanel={<ImageToImageParameters />}>
      <ImageToImageContent />
    </InvokeWorkarea>
  );
}
