package com.example.foodlist.naver.dto;

import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@NoArgsConstructor
public class SearchLocalRes {
    private String lastBuildDate;
    private Integer total;
    private Integer start;
    private Integer display;
    private List<SearchLocalItem> items;


    @Data
    @NoArgsConstructor
    public static class SearchLocalItem{
        private String title;
        private String link;
        private String category;
        private String description;
        private String telephone;
        private String address;
        private String roadAddress;
        private String mapx;
        private String mapy;
    }
}


