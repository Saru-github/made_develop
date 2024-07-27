package com.example.foodlist.db;

import com.example.foodlist.food.entity.Food;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class MemoryDbRepositoryAbstract<T extends MemoryDbEntity> implements MemoryDbRepositoryIfs<T> {
    private final List<T> db = new ArrayList<>();
    private int index = 0;

    @Override
    public Optional<T> findById(int index) {

        try{
            var food = db.stream().filter(it -> it.getIndex() == index).findFirst();
            return food;
        }catch (NullPointerException e){
            return Optional.empty();
        }
    }

    @Override
    public T save(T entity) {
        var food = db.stream().filter(it -> it.getIndex() == entity.getIndex()).findFirst();
        if(food.isPresent()){
            var prevIndex = food.get().getIndex();
            entity.setIndex(prevIndex);
            deleteById(prevIndex);
            db.add(entity);
            return entity;
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
    public List<T> findAll() {
        return db;
    }
}
