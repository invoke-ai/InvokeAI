import { Progress } from '@chakra-ui/react';
import { useAppSelector } from '../../app/hooks';
import { RootState } from '../../app/store';

const ProgressBar = () => {
    const { realSteps } = useAppSelector((state: RootState) => state.sd);
    const { currentStep } = useAppSelector((state: RootState) => state.system);
    const progress = Math.round((currentStep * 100) / realSteps);
    return (
        <Progress size='xs' value={progress} isIndeterminate={progress < 0} />
    );
};

export default ProgressBar;
