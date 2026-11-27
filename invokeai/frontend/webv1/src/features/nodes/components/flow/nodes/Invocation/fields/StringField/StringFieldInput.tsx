import { Input } from '@invoke-ai/ui-library';
import type { FieldComponentProps } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/types';
import { useStringField } from 'features/nodes/components/flow/nodes/Invocation/fields/StringField/useStringField';
import { NO_DRAG_CLASS, NO_PAN_CLASS, NO_WHEEL_CLASS } from 'features/nodes/types/constants';
import type { StringFieldInputInstance, StringFieldInputTemplate } from 'features/nodes/types/field';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

export const StringFieldInput = memo(
  (props: FieldComponentProps<StringFieldInputInstance, StringFieldInputTemplate>) => {
    const { value, onChange } = useStringField(props);
    const { t } = useTranslation();

    return (
      <Input
        className={`${NO_DRAG_CLASS} ${NO_PAN_CLASS} ${NO_WHEEL_CLASS}`}
        placeholder={t('workflows.emptyStringPlaceholder')}
        value={value}
        onChange={onChange}
      />
    );
  }
);

StringFieldInput.displayName = 'StringFieldInput';
