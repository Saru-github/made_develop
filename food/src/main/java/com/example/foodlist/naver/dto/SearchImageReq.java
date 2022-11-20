package com.example.foodlist.naver.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;

@Data
@Builder
public class SearchImageReq {

    @AllArgsConstructor
    public enum ImageFilter {
        ALL("all"),
        LARGE("large"),
        MEDIUM("medium"),
        SMALL("small")
        ;
        private String value;
    }

    @Builder.Default
    private String query = "";

    @Builder.Default
    private int display = 5;

    @Builder.Default
    private int start = 1;

    @Builder.Default
    private String sort = "sim";

    @Builder.Default
    private ImageFilter filter = ImageFilter.ALL;

    public MultiValueMap<String, String> toMultiValueMap(){
        var multiValueMap = new LinkedMultiValueMap<String,String>();
        multiValueMap.add("query", query);
        multiValueMap.add("display", String.valueOf(display));
        multiValueMap.add("start", String.valueOf(start));
        multiValueMap.add("sort", sort);
        multiValueMap.add("filter", filter.value);
        return multiValueMap;
    }
}
