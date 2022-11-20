package com.example.foodlist.naver;

import com.example.foodlist.naver.dto.SearchImageReq;
import com.example.foodlist.naver.dto.SearchLocalReq;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
public class NaverClientTest {

    @Autowired
    private NaverClient naverClient;

    @Test
    public void searchLocalReqBuilderDefaultTest(){
        var req = SearchLocalReq.builder().build();
        Assertions.assertEquals(req.getQuery(), "");
        Assertions.assertEquals(req.getSort(), "random");
        Assertions.assertEquals(req.getDisplay(), 1);
        Assertions.assertEquals(req.getStart(), 1);
    }

    @Test
    public void searchLocalTest(){
        var req = SearchLocalReq.builder().query("모란역 맛집").build();
        var res = naverClient.searchLocal(req);

        Assertions.assertNotNull(res);
        Assertions.assertNotNull(res.getItems());
        Assertions.assertTrue(res.getItems().size() > 0);
    }


    @Test
    public void searchImageBuilderDefaultTest(){
        var req = SearchImageReq.builder().build();

        Assertions.assertEquals(req.getQuery(), "");
        Assertions.assertEquals(req.getSort(), "sim");
        Assertions.assertEquals(req.getDisplay(), 5);
        Assertions.assertEquals(req.getStart(), 1);
        Assertions.assertEquals(req.getFilter(), SearchImageReq.ImageFilter.ALL);
    }

    @Test
    public void searchImageTest(){
        var req = SearchImageReq.builder().query("모란역 맛집").build();
        var res = naverClient.searchImage(req);

        Assertions.assertNotNull(res);
        Assertions.assertNotNull(res.getItems());
        Assertions.assertTrue(res.getItems().size() > 0);
    }
}
