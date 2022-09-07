import { Button } from '@chakra-ui/react';
import { MouseEventHandler } from 'react';

type Props = {
    label: string;
    type?: 'button' | 'submit' | 'reset' | undefined;
    onClick?: MouseEventHandler<HTMLButtonElement>;
    colorScheme?: string;
    isDisabled?: boolean;
    isLoading?: boolean;
};

const SDButton = ({
    label,
    type,
    onClick,
    colorScheme,
    isDisabled = false,
    isLoading = false,
}: Props) => {
    return (
        <Button
            size={'sm'}
            colorScheme={colorScheme}
            onClick={onClick}
            type={type}
            isDisabled={isDisabled}
            isLoading={isLoading}
        >
            {label}
        </Button>
    );
};

export default SDButton;
