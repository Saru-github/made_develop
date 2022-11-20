package com.example.foodlist.naver.dto;

import lombok.Builder;
import lombok.Data;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;

@Data
@Builder
public class SearchLocalReq {
    @Builder.Default
    private String query = "";

    @Builder.Default
    private int display = 1;

    @Builder.Default
    private int start = 1;

    @Builder.Default
    private String sort = "random";

    public MultiValueMap<String, String> toMultiValueMap(){
        var multiValueMap = new LinkedMultiValueMap<String,String>();
        multiValueMap.add("query", query);
        multiValueMap.add("display", String.valueOf(display));
        multiValueMap.add("start", String.valueOf(start));
        multiValueMap.add("sort", sort);
        return multiValueMap;
    }
}
