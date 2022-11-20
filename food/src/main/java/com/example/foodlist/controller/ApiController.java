package com.example.foodlist.controller;

import com.example.foodlist.food.dto.FoodRequestDto;
import com.example.foodlist.food.dto.FoodResultDto;
import com.example.foodlist.food.entity.Food;
import com.example.foodlist.food.repository.FoodRepository;
import com.example.foodlist.food.service.FoodService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Slf4j
@RestController
@RequestMapping("/api/food")
@RequiredArgsConstructor
public class ApiController {

    private final FoodService foodService;
    private final FoodRepository foodRepository;

    @GetMapping("/search")
    public FoodResultDto search(@RequestParam String query){
        return foodService.search(query);
    }

    @PostMapping("")
    public ResponseEntity<FoodResultDto> add(@RequestBody FoodRequestDto foodRequestDto){
        log.info("{}", foodRequestDto);

        var food = new Food();
        food.setAddress(foodRequestDto.getAddress());
        food.setRoadAddress(foodRequestDto.getRoadAddress());
        food.setTitle(foodRequestDto.getTitle());
        food.setCategory(foodRequestDto.getCategory());
        food.setImageLink(foodRequestDto.getImage());
        food.setHomePageLink(foodRequestDto.getLink());
        food.setVisit(false);
        food.setVisitCount(0);
        food.setLastVisitDate(null);
        foodRepository.save(food);
        return ResponseEntity.ok(null);
    }

    @GetMapping("/wish-list")
    public List<Food> wishList(){
        return foodRepository.findAll();
    }

    @PostMapping("/{index}")
    public void addVisit(@PathVariable int index){
        var food = foodRepository.findById(index);
        food.ifPresent(it -> {
            it.setVisit(true);
            it.setVisitCount(it.getVisitCount()+1);
        });
    }

    @DeleteMapping("/{index}")
    public void delete(@PathVariable int index){
        foodRepository.deleteById(index);
    }
}
