import InvokeWorkarea from 'features/ui/components/InvokeWorkarea';
import LinearContent from './LinearContent';
import LinearParameters from './LinearParameters';

export default function LinearWorkarea() {
  return (
    <InvokeWorkarea parametersPanelContent={<LinearParameters />}>
      <LinearContent />
    </InvokeWorkarea>
  );
}
