package com.example.foodlist.food.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@NoArgsConstructor
@AllArgsConstructor
@Data
@Builder
public class FoodRequestDto {
    private String title;
    private String category;
    private String address;
    private String roadAddress;
    private String link;
    private String image;
}
