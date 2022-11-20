package com.example.foodlist.food.entity;

import com.example.foodlist.db.MemoryDbEntity;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@NoArgsConstructor
@AllArgsConstructor
@Data
public class Food extends MemoryDbEntity {

    private String title;                   // 장소
    private String category;                // 카테고리
    private String address;                 // 주소
    private String roadAddress;             // 도로명 주소
    private String homePageLink;            // 홈페이지 링크
    private String imageLink;               // 이미지 링크
    private int imageHeight;                // 이미지 높이
    private int imageWidth;                 // 이미지 넓이
    private boolean isVisit;                // 방문여부
    private LocalDateTime lastVisitDate;    // 마지막 방문일자
    private int visitCount;                 // 방문 횟수
}
