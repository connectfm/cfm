package com.tasks.apmlify;


import android.annotation.SuppressLint;
import android.content.Context;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.work.ListenableWorker;
import androidx.work.Worker;
import androidx.work.WorkerParameters;

import com.amplifyframework.core.Amplify;
import com.amplifyframework.datastore.generated.model.User;

import org.jetbrains.annotations.NotNull;

public class SaveWorker extends Worker {


    public SaveWorker(@NonNull @NotNull Context context, @NonNull @NotNull WorkerParameters workerParams) {
        super(context, workerParams);
    }

    @SuppressLint("RestrictedApi")
    @NonNull
    @NotNull
    @Override
    public Result doWork() {
        User user = User.builder()
                .email("ethan@robvoss.com")
                .build();

        Amplify.DataStore.save(user,
                success -> Log.i("tests", "Saved user " + user.getId()),
                fail -> Log.i("tests", "Failed to save user " + user.getId()));

        return new Result.Success();

    }
}
