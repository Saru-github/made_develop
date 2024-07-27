package com.example.foodlist.naver;

import com.example.foodlist.naver.dto.SearchImageReq;
import com.example.foodlist.naver.dto.SearchLocalReq;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
public class NaverApiServiceTest {

    @Autowired
    private NaverApiService naverApiService;

    @Test
    public void searchImageTest(){
        var req = SearchLocalReq.builder().query("카카오").build();
        var res = naverApiService.searchLocalResOptional(req);

        Assertions.assertTrue(res.isPresent());
    }

    @Test
    public void searchLocalTest(){
        var req = SearchImageReq.builder().query("카카오").build();
        var res = naverApiService.searchImageResOptional(req);

        Assertions.assertTrue(res.isPresent());
    }
}
