package com.example.foodlist.db;

import com.example.foodlist.food.entity.Food;
import com.example.foodlist.food.repository.FoodRepository;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
public class FoodRepositoryTest {

    @Autowired
    private FoodRepository foodRepository;

    @Test
    public void createTest(){
        var food = createObj("");
        var expected = foodRepository.save(food);

        Assertions.assertNotNull(expected);
        Assertions.assertEquals(1,expected.getIndex());
    }

    @Test
    public void readTest(){
        var food = createObj("");
        foodRepository.save(food);

        var expected = foodRepository.findById(1);
        Assertions.assertNotNull(expected);
    }

    @Test
    public void updateTest(){
        var food = createObj("");
        var saveEntity = foodRepository.save(food);

        saveEntity.setVisit(true);
        var expected = foodRepository.save(saveEntity);

        Assertions.assertNotNull(expected);
        Assertions.assertEquals(true, expected.isVisit());
    }

    @Test
    public void deleteTest(){
        var food = createObj("");
        foodRepository.save(food);

        foodRepository.deleteById(1);
        var expected = foodRepository.findById(1);

        Assertions.assertEquals(false, expected.isPresent());
    }

    private Food createObj(String prefix){
        var food = new Food();
        food.setTitle(prefix+" title");
        food.setCategory(prefix+" category");
        food.setAddress(prefix+" address");
        food.setRoadAddress(prefix+" read address");
        food.setHomePageLink(prefix+" home page");
        food.setImageLink(prefix+" image link");
        food.setImageWidth(10);
        food.setImageHeight(10);
        food.setVisit(false);
        food.setLastVisitDate(null);
        food.setVisitCount(0);
        return food;
    }
}
