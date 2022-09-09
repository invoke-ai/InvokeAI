import { Button, ButtonProps } from '@chakra-ui/react';

interface Props extends ButtonProps {
    label: string;
}

const SDButton = (props: Props) => {
    const { label, size = 'sm' } = props;
    return (
        <Button size={size} {...props}>
            {label}
        </Button>
    );
};

export default SDButton;
