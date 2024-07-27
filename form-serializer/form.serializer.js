(function (root) {
    'use strict';

    function serialize(form) {
        var result = {};

        if (form === null) {
            return result;
        }

        var serializeOptions = extractSerializeOptions(form);
        var formElements = Array.from(form.elements);

        formElements.forEach(function (elem) {
            if (!elem.name || elem.disabled || elem.type === 'submit' || elem.type === 'button' || elem.matches('form fieldset[disabled] *')) {
                return;
            }
            if (elem.type === 'select-one') {
                result[elem.name] = elem.value;
            } else if (elem.type === 'select-multiple') {
                var multipleValues = [];
                var selectOptions = Array.from(elem.options);
                selectOptions.forEach(function (opt) {
                    if (!opt.disabled && opt.selected) {
                        multipleValues.push(opt.value);
                    }
                });

                result[elem.name] = multipleValues;

            } else if (elem.type === 'checkbox' || elem.type === 'radio') {
                if (elem.checked) {
                    result[elem.name] = elem.value;
                }
            } else if (elem.type === 'textarea') {
                result[elem.name] = elem.value.replace(/\r?\n|\r/g, '\n');
            } else {
                result[elem.name] = (elem.value ? elem.value.trim() : elem.value);
            }

            // 옵션 처리 시작
            if (result[elem.name] === undefined) {
                return result;
            }

            var delimiterOptionData = serializeOptions[elem.name].delimiterOption;
            var caseOptionData = serializeOptions[elem.name].case;
            // delimiter 옵션 (delimiter, max)
            if (delimiterOptionData !== undefined) {
                result[elem.name] = checkDelimiterOption(delimiterOptionData, result[elem.name]);
            }

            // case 옵션
            if (caseOptionData !== undefined) {
                result[elem.name] = checkCaseOption(caseOptionData, result[elem.name]);
            }
        });

        return result;
    }

    function extractSerializeOptions(form) {
        var serializeOptions = {};
        var formElements = Array.from(form.elements);

        var delimiterValue = {
            COMMA: ",",
            LINE: "\n"
        }

        formElements.forEach(function (elem) {
            serializeOptions[elem.name] = {};
            var delimiterData = elem.dataset.serializeDelimiter;
            var maxData = elem.dataset.serializeMax;
            var caseData = elem.dataset.serializeCase;
            if (delimiterData !== undefined) {
                serializeOptions[elem.name].delimiterOption = {};
                serializeOptions[elem.name].delimiterOption.delimiter = delimiterValue[delimiterData.toUpperCase()];
                if (maxData !== undefined) {
                    serializeOptions[elem.name].delimiterOption.max = maxData;
                }
            }
            if (caseData !== undefined) {
                serializeOptions[elem.name].case = caseData;
            }
        });

        return serializeOptions;
    }

    var checkDelimiterOption = (option, result) => {
        result = result.split(option.delimiter)
            .map(value => value.replace('/[\x00-\x1F]/g', '').trim())
            .filter(value => value !== '');

        if (option.max !== undefined) {
            var valuesLength = result.length;
            if (valuesLength && option.max > 0) {
                if (valuesLength > option.max) {
                    alert('복수행 조회 조건은 ' + option.max + '행을 넘을 수 없습니다. [입력 행: ' + valuesLength + '행]');
                    return false;
                }
            }
        }

        return result
    }

    var checkCaseOption = (option, result) => {
        if (Array.isArray(result)) {
            if (option === "upper") {
                result = result.map(value => value.toUpperCase());
            } else if (option === "lower") {
                result = result.map(value => value.toLowerCase());
            }
        } else {
            if (typeof result === "string"){
                if (option === 'upper') {
                    result = result.toUpperCase();
                } else if (option === 'lower') {
                    result = result.toLowerCase();
                }
            }
        }

        return result
    }

    window.FormSerializer = {
        serialize: serialize
    };
})(window);
