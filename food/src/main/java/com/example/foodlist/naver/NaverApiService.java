package com.example.foodlist.naver;

import com.example.foodlist.naver.dto.SearchImageReq;
import com.example.foodlist.naver.dto.SearchImageRes;
import com.example.foodlist.naver.dto.SearchLocalReq;
import com.example.foodlist.naver.dto.SearchLocalRes;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.Optional;

@Slf4j
@Service
@RequiredArgsConstructor
public class NaverApiService {

    private final NaverClient naverClient;

    public Optional<SearchLocalRes> searchLocalResOptional(SearchLocalReq searchLocalReq){
        try{
            var res = naverClient.searchLocal(searchLocalReq);
            return Optional.ofNullable(res);
        }catch (Exception e){
            log.error("searchLocalResOptional Error ",e);
            throw e;
        }
    }

    public Optional<SearchImageRes> searchImageResOptional(SearchImageReq searchImageReq){
        try{
            var res = naverClient.searchImage(searchImageReq);
            return Optional.ofNullable(res);
        }catch (Exception e){
            log.error("searchImageResOptional Error ",e);
            throw e;
        }
    }
}
