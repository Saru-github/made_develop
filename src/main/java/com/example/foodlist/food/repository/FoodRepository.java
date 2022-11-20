package com.example.foodlist.food.repository;

import com.example.foodlist.db.MemoryDbRepositoryAbstract;
import com.example.foodlist.food.entity.Food;
import org.springframework.stereotype.Repository;

@Repository
public class FoodRepository extends MemoryDbRepositoryAbstract<Food> {

}
