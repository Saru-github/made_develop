package com.example.foodlist.food.service;

import com.example.foodlist.food.dto.FoodResultDto;
import com.example.foodlist.naver.NaverApiService;
import com.example.foodlist.naver.dto.SearchImageReq;
import com.example.foodlist.naver.dto.SearchLocalReq;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

@Slf4j
@Service
@RequiredArgsConstructor
public class FoodService {

    private final NaverApiService naverApiService;

    public FoodResultDto search(String query){
        var searchLocalReq = SearchLocalReq.builder().query(query).build();
        var localResOptional = naverApiService.searchLocalResOptional(searchLocalReq);

        if(localResOptional.isPresent() && localResOptional.get().getItems().stream().findFirst().isPresent()){
            var location = localResOptional.get().getItems().stream().findFirst().get();
            var imageQuery = location.getTitle().replaceAll("<[^>]*>", "");
            var searchImageReq = SearchImageReq
                .builder()
                .query(imageQuery)
                .build();

            var imageOptional = naverApiService.searchImageResOptional(searchImageReq);

            if(imageOptional.isPresent() && imageOptional.get().getItems().stream().findFirst().isPresent()) {
                var image = imageOptional.get().getItems().stream().findFirst().get();
                return FoodResultDto.builder()
                        .title(location.getTitle())
                        .address(location.getAddress())
                        .roadAddress(location.getRoadAddress())
                        .category(location.getCategory())
                        .link(location.getLink())
                        .image(image.getLink())
                        .build();
            }
            return FoodResultDto.builder().build();
        }

        return FoodResultDto.builder().build();
    }
}
