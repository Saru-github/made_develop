package com.example.foodlist.food.repository;

import com.example.foodlist.db.MemoryDbRepositoryIfs;
import com.example.foodlist.food.entity.Food;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class FoodRepositoryAbstract implements MemoryDbRepositoryIfs<Food> {
    private final List<Food> db = new ArrayList<>();
    private int index = 0;

    @Override
    public Optional<Food> findById(int index) {
        try{
            var food = db.stream().filter(it -> it.getIndex() == index).findFirst();
            return food;
        }catch (NullPointerException e){
            return Optional.empty();
        }
    }

    @Override
    public Food save(Food entity) {
        var food = db.stream().filter(it -> it.getIndex() == index).findFirst();
        if(food.isPresent()){
            var it = food.get();
            it.setTitle(entity.getTitle());
            it.setCategory(entity.getCategory());
            it.setAddress(entity.getAddress());
            it.setRoadAddress(entity.getRoadAddress());
            it.setHomePageLink(entity.getHomePageLink());
            it.setImageLink(entity.getImageLink());
            it.setImageWidth(entity.getImageWidth());
            it.setImageHeight(entity.getImageHeight());
            it.setVisit(entity.isVisit());
            it.setLastVisitDate(entity.getLastVisitDate());
            it.setVisitCount(entity.getVisitCount());
            return it;
        }else{
            index++;
            entity.setIndex(index);
            db.add(entity);
            return entity;
        }
    }

    @Override
    public void deleteById(int index) {
        var optionalFood = db.stream().filter(it -> it.getIndex() == index).findFirst();
        optionalFood.ifPresent(db::remove);
    }

    @Override
    public List<Food> findAll() {
        return db;
    }
}
