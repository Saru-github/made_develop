package com.example.foodlist.naver.dto;

import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@NoArgsConstructor
public class SearchImageRes {
    private String lastBuildDate;
    private Integer total;
    private Integer start;
    private Integer display;
    private List<SearchImageItem> items;

    @Data
    @NoArgsConstructor
    public static class SearchImageItem{
        private String title;
        private String link;
        private String thumbnail;
        private String sizeheight;
        private String sizewidth;
    }
}


